# services/improved_rag_engine.py - Enhanced RAG with Gemini Embeddings

import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient
import google.generativeai as genai
import json
import re
import time

load_dotenv()
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Enhanced RAG Engine with:
    - Gemini API embeddings (lightweight, no local models)
    - Better prompt engineering
    - Multi-aspect violation detection
    - Improved JSON parsing
    """

    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.model_name = "gemini-2.0-flash"
        self.embedding_model = "models/text-embedding-004"  # Latest Gemini embedding model
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
        
        logger.info(f"RAG Engine initialized with database: {db_name}")
        logger.info("✅ Using Gemini embeddings - No heavy ML libraries needed!")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector using Gemini API.
        Maintains same function signature as before.
        """
        try:
            # Use Gemini's embedding API
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            
            # Add small delay to respect rate limits (15/min)
            time.sleep(0.05)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            # Retry once after a delay
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
        """Calculate cosine similarity between vectors."""
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
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant rules using vector similarity."""
        
        # Query for the policy document (not individual rules)
        policy_doc = self.rules_collection.find_one({"company": company, "status": "active"})
        
        if not policy_doc:
            # Try without status filter
            policy_doc = self.rules_collection.find_one({"company": company})
        
        if not policy_doc:
            logger.warning(f"No policy document found for company={company}")
            return []
        
        # Extract rules from the nested rules_extracted array
        all_rules = policy_doc.get('rules_extracted', [])
        
        if not all_rules:
            logger.warning(f"No rules found in policy document for company={company}")
            return []
        
        logger.info(f"Found {len(all_rules)} total rules in policy document")
        
        # Optional category filter
        bill_category = bill_facts.get('category', bill_facts.get('bill_meta', {}).get('category'))
        
        if bill_category:
            # Filter rules by category (including "Other" category)
            filtered_rules = [
                rule for rule in all_rules 
                if rule.get('category') in [bill_category, 'Other']
            ]
            
            if filtered_rules:
                logger.info(f"Filtered to {len(filtered_rules)} rules matching category '{bill_category}' or 'Other'")
                all_rules = filtered_rules
            else:
                logger.warning(f"No rules found for category={bill_category}, using all rules")

        # Score by similarity
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
            # Return rules without scoring
            logger.warning("No rules with valid embeddings, returning first rules")
            return all_rules[:top_k]

        # Sort by similarity
        scored_rules.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Also include high-severity rules regardless of similarity
        high_severity_rules = [r for r in all_rules if r.get('severity') == 'HIGH']
        
        # Merge and deduplicate by rule_id
        top_rules = scored_rules[:top_k]
        seen_rule_ids = {rule.get('rule_id') for rule in top_rules}
        
        for hs_rule in high_severity_rules:
            if hs_rule.get('rule_id') not in seen_rule_ids and len(top_rules) < top_k * 2:
                top_rules.append(hs_rule)
                seen_rule_ids.add(hs_rule.get('rule_id'))
        
        logger.info(f"Returning {len(top_rules)} relevant rules")
        return top_rules[:top_k * 2]  # Return up to 2x top_k to ensure coverage

    def reason_with_llm(
        self,
        bill_facts: Dict[str, Any],
        policy_rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced LLM reasoning with comprehensive violation detection.
        """
        
        bill_meta = bill_facts.get('bill_meta', {})
        
        # Construct detailed bill description
        bill_description = self._format_bill_details(bill_facts)
        
        # Construct rule description with all attributes
        rule_description = self._format_rule_details(policy_rule)
        
        prompt = f"""You are an expert compliance auditor analyzing expense claims against company policies.

YOUR TASK: Analyze if this expense bill complies with the given policy rule. Be thorough and check ALL aspects.

POLICY RULE:
{rule_description}

EXPENSE BILL:
{bill_description}

ANALYSIS INSTRUCTIONS:
1. Check EVERY aspect of the rule against the bill
2. Look for violations in: amounts, modes/types, conditions, approvals, documentation, restrictions
3. Be strict - any deviation from policy is a violation
4. Consider context - sometimes bills may be compliant even if they seem borderline
5. Provide specific, actionable explanations

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "compliant": true or false,
  "classification": "Compliant" OR one of ["Exceeded Limit", "Disallowed Mode", "Missing Approval", "Missing Documentation", "Unauthorized Vendor", "Prohibited Category", "Time Restriction Violated", "Quantity Exceeded", "Grade Restriction", "Geographic Restriction", "Not Covered", "Other Violation"],
  "severity": "HIGH" (critical violation) OR "MEDIUM" (moderate issue) OR "LOW" (minor concern),
  "explanation": "Clear, specific explanation citing exact policy text and bill details. Explain WHAT is wrong, WHY it violates policy, and HOW much/by what degree.",
  "confidence": 0.0 to 1.0 (your confidence in this assessment),
  "violation_details": {{
    "expected": "what the policy requires/allows",
    "actual": "what the bill shows",
    "deviation": "specific deviation amount/type if applicable"
  }}
}}

IMPORTANT RULES:
- If compliant, set "compliant": true and "classification": "Compliant"
- If ANY aspect violates policy, set "compliant": false
- Use the most specific classification that matches the violation
- Higher severity for clear, significant violations; lower for ambiguous cases
- Be precise with numbers - state exact amounts and limits
- If rule doesn't apply to this bill (wrong category/context), mark as compliant

Return ONLY valid JSON, no other text.
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
            
            result = json.loads(result_json)
            
            # Validate result structure
            result = self._validate_llm_result(result)
            
            # Add metadata
            result['company_rule_text'] = policy_rule.get('raw_text', '')
            result['rule_id'] = policy_rule.get('rule_id', '')
            result['rule_category'] = policy_rule.get('category', '')
            result['bill_snippet'] = bill_facts.get('raw_text', '')[:200]
            result['model_used'] = self.model_name
            
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {result_text[:500]}")
            return self._fallback_result(bill_facts, policy_rule, f"JSON parsing error: {str(e)}")
        
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            return self._fallback_result(bill_facts, policy_rule, f"Analysis error: {str(e)}")

    def _format_bill_details(self, bill_facts: Dict[str, Any]) -> str:
        """Format bill details for LLM prompt."""
        bill_meta = bill_facts.get('bill_meta', {})
        
        lines = []
        lines.append(f"Category: {bill_meta.get('category', 'N/A')}")
        lines.append(f"Amount: {bill_meta.get('amount', 'N/A')} {bill_meta.get('currency', 'INR')}")
        
        if bill_meta.get('mode'):
            lines.append(f"Mode/Type: {bill_meta.get('mode')}")
        if bill_meta.get('vendor'):
            lines.append(f"Vendor: {bill_meta.get('vendor')}")
        if bill_meta.get('date'):
            lines.append(f"Date: {bill_meta.get('date')}")
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
        
        # Add raw text if available
        if bill_facts.get('raw_text'):
            lines.append(f"\nOriginal Bill Text:\n{bill_facts.get('raw_text')[:300]}")
        
        return "\n".join(lines)

    def _format_rule_details(self, policy_rule: Dict[str, Any]) -> str:
        """Format policy rule details for LLM prompt."""
        lines = []
        
        lines.append(f"Rule ID: {policy_rule.get('rule_id', 'N/A')}")
        lines.append(f"Category: {policy_rule.get('category', 'N/A')}")
        lines.append(f"Severity: {policy_rule.get('severity', 'MEDIUM')}")
        lines.append(f"\nRule Text:\n{policy_rule.get('raw_text', 'N/A')}")
        
        # Add structured attributes
        attrs = policy_rule.get('attributes', {})
        if attrs:
            lines.append("\nRule Attributes:")
            if attrs.get('max_amount'):
                lines.append(f"  • Max Amount: {attrs.get('max_amount')} {attrs.get('currency', 'INR')}")
            if attrs.get('min_amount'):
                lines.append(f"  • Min Amount: {attrs.get('min_amount')} {attrs.get('currency', 'INR')}")
            if attrs.get('allowed_modes'):
                lines.append(f"  • Allowed Modes: {', '.join(attrs.get('allowed_modes'))}")
            if attrs.get('disallowed_modes'):
                lines.append(f"  • Disallowed Modes: {', '.join(attrs.get('disallowed_modes'))}")
            if attrs.get('conditions'):
                lines.append(f"  • Conditions: {', '.join(attrs.get('conditions'))}")
            if attrs.get('domestic_only'):
                lines.append(f"  • Domestic Only: Yes")
            if attrs.get('scope'):
                lines.append(f"  • Scope: {attrs.get('scope')}")
            if attrs.get('vendor_restrictions'):
                lines.append(f"  • Approved Vendors: {', '.join(attrs.get('vendor_restrictions'))}")
        
        return "\n".join(lines)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response with various formats."""
        # Try direct JSON parse
        try:
            json.loads(text)
            return text
        except:
            pass
        
        # Extract from markdown code blocks
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
            return text
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
            return text
        
        # Try to find JSON object with regex
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        return None

    def _validate_llm_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix LLM result structure."""
        # Ensure required fields
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
        
        # Ensure confidence is between 0 and 1
        confidence = result.get('confidence', 0.7)
        if isinstance(confidence, (int, float)):
            result['confidence'] = max(0.0, min(1.0, float(confidence)))
        else:
            result['confidence'] = 0.7
        
        # Normalize severity
        severity = str(result.get('severity', 'MEDIUM')).upper()
        if severity not in ['HIGH', 'MEDIUM', 'LOW']:
            result['severity'] = 'MEDIUM'
        
        return result

    def _fallback_result(
        self,
        bill_facts: Dict[str, Any],
        policy_rule: Dict[str, Any],
        error_msg: str
    ) -> Dict[str, Any]:
        """Return fallback result when LLM fails."""
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