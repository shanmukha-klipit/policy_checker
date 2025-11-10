# services/optimized_rag_engine.py - Optimized RAG with Batched Rule Checking

import os
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient
import google.generativeai as genai
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Optimized RAG Engine with:
    - Batched rule checking (all rules in one LLM call)
    - Parallel embedding generation
    - Faster response times
    - Same API interface as before
    """

    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.model_name = "gemini-2.0-flash"
        self.embedding_model = "models/text-embedding-004"
        logger.info(f"Using LLM model: {self.model_name}")
        logger.info(f"Using embedding model: {self.embedding_model}")

        # MongoDB - Read from environment variables
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI not set in environment variables")
        
        db_name = os.getenv("MONGODB_DB_NAME")
        
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.rules_collection = self.db['policy_rules']
        
        logger.info(f"Optimized RAG Engine initialized with database: {db_name}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using Gemini API (unchanged for compatibility)"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            time.sleep(0.05)  # Rate limiting
            
            return embedding
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            try:
                time.sleep(1)
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as retry_error:
                logger.error(f"Embedding retry failed: {retry_error}")
                raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors (unchanged)"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def retrieve_relevant_rules(
        self,
        company: str,
        bill_embedding: List[float],
        bill_facts: Dict[str, Any],
        top_k: int = 10,
        policy_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant rules using vector similarity (unchanged logic)"""
        try:
            if policy_name:
                logger.info(f"Fetching rules for company={company}, policy_name={policy_name}")
                policy_docs = list(self.rules_collection.find({
                    "company": company,
                    "policy_name": policy_name,
                    "status": "active"
                }))
                
                if not policy_docs:
                    policy_docs = list(self.rules_collection.find({
                        "company": company,
                        "policy_name": policy_name
                    }))
            else:
                logger.info(f"Fetching rules for all active policies of company={company}")
                policy_docs = list(self.rules_collection.find({
                    "company": company,
                    "status": "active"
                }))
                
                if not policy_docs:
                    policy_docs = list(self.rules_collection.find({
                        "company": company
                    }))
            
            if not policy_docs:
                logger.warning(f"No policy document(s) found for company={company}")
                return []

            all_rules = []
            for doc in policy_docs:
                rules = doc.get('rules_extracted', [])
                if rules:
                    all_rules.extend(rules)

            if not all_rules:
                logger.warning(f"No rules found in policy document(s) for company={company}")
                return []

            logger.info(f"Collected total {len(all_rules)} rules from {len(policy_docs)} policy document(s)")

            bill_category = bill_facts.get('category', bill_facts.get('bill_meta', {}).get('category'))
            if bill_category:
                filtered_rules = [
                    rule for rule in all_rules
                    if rule.get('category') in [bill_category, 'Other']
                ]
                if filtered_rules:
                    logger.info(f"Filtered to {len(filtered_rules)} rules matching category '{bill_category}' or 'Other'")
                    all_rules = filtered_rules
                else:
                    logger.warning(f"No rules found for category={bill_category}, using all rules")

            scored_rules = []
            for rule in all_rules:
                if 'embedding' in rule and rule['embedding']:
                    try:
                        similarity = self.cosine_similarity(bill_embedding, rule['embedding'])
                        rule['similarity_score'] = similarity
                        scored_rules.append(rule)
                    except Exception as e:
                        logger.warning(f"Skipping rule {rule.get('rule_id')}: {e}")

            if not scored_rules:
                logger.warning("No rules with valid embeddings, returning first rules")
                return all_rules[:top_k]

            scored_rules.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            high_severity_rules = [r for r in all_rules if r.get('severity') == 'HIGH']

            top_rules = scored_rules[:top_k]
            seen_rule_ids = {rule.get('rule_id') for rule in top_rules}

            for hs_rule in high_severity_rules:
                if hs_rule.get('rule_id') not in seen_rule_ids and len(top_rules) < top_k * 2:
                    top_rules.append(hs_rule)
                    seen_rule_ids.add(hs_rule.get('rule_id'))

            logger.info(f"Returning {len(top_rules)} relevant rules")
            return top_rules[:top_k * 2]

        except Exception as e:
            logger.error(f"Error retrieving rules for {company}: {e}", exc_info=True)
            return []

    def reason_with_llm_batch(
        self,
        bill_facts: Dict[str, Any],
        policy_rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        🚀 OPTIMIZED: Check ALL rules in a SINGLE LLM call instead of sequential calls.
        This is the main optimization that reduces time from N API calls to 1 API call.
        """
        
        if not policy_rules:
            return []
        
        bill_meta = bill_facts.get('bill_meta', {})
        
        # Construct detailed bill description
        bill_description = self._format_bill_details(bill_facts)
        
        # Construct all rules in a structured format
        rules_text = self._format_all_rules(policy_rules)
        

        
        prompt = f"""You are an expert compliance auditor analyzing an expense claim against multiple company policy rules.

        YOUR TASK: Analyze if this expense bill complies with ALL the given policy rules. Check each rule thoroughly.

        EXPENSE BILL DETAILS:
        {bill_description}

        POLICY RULES TO CHECK (Total: {len(policy_rules)} rules):
        {rules_text}

        ANALYSIS INSTRUCTIONS:
        1. Check the bill against EVERY rule listed above
        2. For each rule, determine if there's a violation
        3. Look for violations in: amounts, modes/types, conditions, approvals, documentation, restrictions
        4. **CRITICAL FOR TIME RESTRICTIONS**: 
        - Check if bill submission date exceeds policy time limits
        - Calculate days between bill date and current date
        - If policy says "within X days" and bill is older, it's a violation
        - Example: Bill from 2025-06-15, today is 2025-11-10 = 148 days old
        - If policy says "within 30 days", this violates the rule
        5. Be strict - any deviation from policy is a violation
        6. If a rule doesn't apply to this bill (wrong category/context), mark it as compliant
        7. Provide specific, actionable explanations with exact dates and calculations

        RESPOND IN THIS EXACT JSON FORMAT - AN ARRAY OF RESULTS FOR EACH RULE:
        [
        {{
            "rule_id": "r1",
            "compliant": true or false,
            "classification": "Compliant" OR one of ["Exceeded Limit", "Disallowed Mode", "Missing Approval", "Missing Documentation", "Unauthorized Vendor", "Prohibited Category", "Time Restriction Violated", "Quantity Exceeded", "Grade Restriction", "Geographic Restriction", "Not Covered", "Other Violation"],
            "severity": "HIGH" or "MEDIUM" or "LOW",
            "explanation": "Clear, specific explanation citing exact policy text and bill details. For time violations: state bill date, current date, days elapsed, and policy limit.",
            "confidence": 0.0 to 1.0,
            "violation_details": {{
            "expected": "what the policy requires/allows",
            "actual": "what the bill shows",
            "deviation": "specific deviation amount/type/days if applicable"
            }}
        }},
        ... (one object for each rule)
        ]

        IMPORTANT RULES:
        - Return exactly {len(policy_rules)} result objects (one per rule)
        - Include rule_id in each result to identify which rule it refers to
        - If compliant, set "compliant": true and "classification": "Compliant"
        - If ANY aspect violates policy, set "compliant": false
        - For time violations, use classification "Time Restriction Violated"
        - Use the most specific classification that matches the violation
        - Higher severity for clear, significant violations; lower for ambiguous cases
        - Be precise with numbers and dates - state exact amounts, limits, and time calculations

        Return ONLY valid JSON array, no other text.
        """
        
        try:
            model = genai.GenerativeModel(self.model_name)
            
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
            
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                )
            )
            
            result_text = response.text.strip()
            
            # Extract JSON from response
            result_json = self._extract_json(result_text)
            
            if not result_json:
                raise ValueError("Could not extract valid JSON from LLM response")
            
            results = json.loads(result_json)
            
            # Validate it's an array
            if not isinstance(results, list):
                raise ValueError("Expected JSON array of results")
            
            # Enrich each result with metadata
            enriched_results = []
            rule_map = {r.get('rule_id'): r for r in policy_rules}
            
            for result in results:
                result = self._validate_llm_result(result)
                
                # Add rule metadata
                rule_id = result.get('rule_id', '')
                if rule_id in rule_map:
                    rule = rule_map[rule_id]
                    result['company_rule_text'] = rule.get('raw_text', '')
                    result['rule_category'] = rule.get('category', '')
                else:
                    result['company_rule_text'] = ''
                    result['rule_category'] = ''
                
                result['bill_snippet'] = bill_facts.get('raw_text', '')[:200]
                result['model_used'] = self.model_name
                
                enriched_results.append(result)
            
            logger.info(f"Batch analysis complete: {len(enriched_results)} rules checked in single call")
            return enriched_results

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in batch analysis: {e}")
            logger.error(f"Raw response: {result_text[:500]}")
            return self._fallback_batch_results(bill_facts, policy_rules, f"JSON parsing error: {str(e)}")
        
        except Exception as e:
            logger.error(f"LLM batch reasoning error: {e}")
            return self._fallback_batch_results(bill_facts, policy_rules, f"Analysis error: {str(e)}")

    def reason_with_llm(
        self,
        bill_facts: Dict[str, Any],
        policy_rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Kept for backward compatibility, but internally uses batch processing.
        This maintains the same API interface for existing code.
        """
        # Convert single rule to batch and return first result
        results = self.reason_with_llm_batch(bill_facts, [policy_rule])
        return results[0] if results else self._fallback_result(bill_facts, policy_rule, "No results")

    def _format_all_rules(self, policy_rules: List[Dict[str, Any]]) -> str:
        """Format all rules into a compact, structured text format for the prompt."""
        lines = []
        
        for idx, rule in enumerate(policy_rules, 1):
            lines.append(f"\n--- RULE {idx}: {rule.get('rule_id', f'rule_{idx}')} ---")
            lines.append(f"Category: {rule.get('category', 'N/A')}")
            lines.append(f"Severity: {rule.get('severity', 'MEDIUM')}")
            lines.append(f"Rule Text: {rule.get('raw_text', 'N/A')}")
            
            # Add structured attributes in compact form
            attrs = rule.get('attributes', {})
            if attrs:
                attr_parts = []
                if attrs.get('max_amount'):
                    attr_parts.append(f"Max: {attrs.get('max_amount')} {attrs.get('currency', 'INR')}")
                if attrs.get('min_amount'):
                    attr_parts.append(f"Min: {attrs.get('min_amount')} {attrs.get('currency', 'INR')}")
                if attrs.get('allowed_modes'):
                    attr_parts.append(f"Allowed: {', '.join(attrs.get('allowed_modes'))}")
                if attrs.get('disallowed_modes'):
                    attr_parts.append(f"Disallowed: {', '.join(attrs.get('disallowed_modes'))}")
                if attrs.get('conditions'):
                    attr_parts.append(f"Conditions: {', '.join(attrs.get('conditions'))}")
                if attrs.get('domestic_only'):
                    attr_parts.append("Domestic Only")
                
                if attr_parts:
                    lines.append(f"Attributes: {' | '.join(attr_parts)}")
        
        return "\n".join(lines)

    def _format_bill_details(self, bill_facts: Dict[str, Any]) -> str:
        """Format bill details for LLM prompt"""
        bill_meta = bill_facts.get('bill_meta', {})
        
        lines = []
        lines.append(f"Category: {bill_meta.get('category', 'N/A')}")
        lines.append(f"Amount: {bill_meta.get('amount', 'N/A')} {bill_meta.get('currency', 'INR')}")
        
        # IMPORTANT: Include the bill date prominently for time restriction checks
        if bill_meta.get('date'):
            bill_date = bill_meta.get('date')
            lines.append(f"Bill Date: {bill_date}")
            
            # Calculate age of bill for time restriction analysis
            try:
                from datetime import datetime
                if isinstance(bill_date, str):
                    bill_date_obj = datetime.fromisoformat(bill_date.replace('Z', '+00:00'))
                else:
                    bill_date_obj = bill_date
                
                current_date = datetime.utcnow()
                days_old = (current_date - bill_date_obj).days
                
                lines.append(f"Bill Age: {days_old} days old (submitted on {current_date.strftime('%Y-%m-%d')})")
                
                logger.info(f"🕒 Bill Date: {bill_date}, Current: {current_date.strftime('%Y-%m-%d')}, Age: {days_old} days")
                
            except Exception as e:
                logger.warning(f"Could not calculate bill age: {e}")
        
        if bill_meta.get('mode'):
            lines.append(f"Mode/Type: {bill_meta.get('mode')}")
        if bill_meta.get('vendor'):
            lines.append(f"Vendor: {bill_meta.get('vendor')}")
        if bill_meta.get('origin'):
            lines.append(f"Origin: {bill_meta.get('origin')}")
        if bill_meta.get('destination'):
            lines.append(f"Destination: {bill_meta.get('destination')}")
        if bill_meta.get('description'):
            lines.append(f"Description: {bill_meta.get('description')}")
        if bill_meta.get('quantity'):
            lines.append(f"Quantity: {bill_meta.get('quantity')}")
        if bill_meta.get('has_receipt'):
            lines.append(f"Receipt Attached: {bill_meta.get('has_receipt')}")
        if bill_meta.get('approval_status'):
            lines.append(f"Approval Status: {bill_meta.get('approval_status')}")
        
        if bill_facts.get('raw_text'):
            lines.append(f"\nOriginal Bill Text:\n{bill_facts.get('raw_text')[:300]}")
        
        return "\n".join(lines)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response (unchanged)"""
        try:
            json.loads(text)
            return text
        except:
            pass
        
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
            return text
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
            return text
        
        # Try to find JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return match.group(0)
        
        # Try to find JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        return None

    def _validate_llm_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix LLM result structure (unchanged)"""
        if 'compliant' not in result:
            result['compliant'] = False
        
        if 'classification' not in result:
            result['classification'] = "Other Violation"
        
        if 'severity' not in result:
            result['severity'] = "MEDIUM"
        
        if 'explanation' not in result:
            result['explanation'] = "Compliance check completed"
        
        if 'confidence' not in result:
            result['confidence'] = 0.7
        
        confidence = result.get('confidence', 0.7)
        if isinstance(confidence, (int, float)):
            result['confidence'] = max(0.0, min(1.0, float(confidence)))
        else:
            result['confidence'] = 0.7
        
        severity = str(result.get('severity', 'MEDIUM')).upper()
        if severity not in ['HIGH', 'MEDIUM', 'LOW']:
            result['severity'] = 'MEDIUM'
        
        return result

    def _fallback_batch_results(
        self,
        bill_facts: Dict[str, Any],
        policy_rules: List[Dict[str, Any]],
        error_msg: str
    ) -> List[Dict[str, Any]]:
        """Return fallback results for all rules when batch processing fails."""
        results = []
        for rule in policy_rules:
            results.append({
                "rule_id": rule.get('rule_id', ''),
                "compliant": False,
                "classification": "Analysis Error",
                "severity": "LOW",
                "explanation": f"Unable to complete compliance analysis: {error_msg}",
                "confidence": 0.3,
                "company_rule_text": rule.get('raw_text', ''),
                "rule_category": rule.get('category', ''),
                "bill_snippet": bill_facts.get('raw_text', '')[:200],
                "model_used": self.model_name,
                "error": error_msg
            })
        return results

    def _fallback_result(
        self,
        bill_facts: Dict[str, Any],
        policy_rule: Dict[str, Any],
        error_msg: str
    ) -> Dict[str, Any]:
        """Return fallback result when single LLM call fails (unchanged)"""
        return {
            "compliant": False,
            "classification": "Analysis Error",
            "severity": "LOW",
            "explanation": f"Unable to complete compliance analysis: {error_msg}",
            "confidence": 0.3,
            "company_rule_text": policy_rule.get('raw_text', ''),
            "rule_id": policy_rule.get('rule_id', ''),
            "rule_category": policy_rule.get('category', ''),
            "bill_snippet": bill_facts.get('raw_text', '')[:200],
            "model_used": self.model_name,
            "error": error_msg
        }