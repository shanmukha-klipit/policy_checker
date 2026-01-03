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
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import random  # For exponential backoff

load_dotenv()
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    RAG Engine with static MongoDB connection (no dynamic origin-based switching)
    """
    
    def __init__(self, origin: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables")
        
        genai.configure(api_key=api_key)
        self.model_name = "gemini-2.0-flash"
        self.embedding_model = "text-embedding-004"
        
        logger.info(f"Using LLM model: {self.model_name}")
        logger.info(f"Using embedding model: {self.embedding_model}")

        self._initialize_static_connection()
        
        logger.info(f"✅ RAG Engine initialized with static MongoDB connection")

    def _initialize_static_connection(self):
        """Initialize MongoDB connection with default settings"""
        default_mongo_uri = os.getenv("MONGODB_URI")
        default_db_name = os.getenv("MONGODB_DB_NAME", "klipit")
        
        try:
            self.mongo_client = MongoClient(default_mongo_uri)
            self.db = self.mongo_client[default_db_name]
            self.rules_collection = self.db['policy_rules']
            
            logger.info(f"✅ RAG Engine connected to MongoDB")
            logger.info(f"   Database: {default_db_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Engine connection: {e}")
            raise

    def generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding with exponential backoff and retry logic.
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for embedding, returning zero vector")
            return [0.0] * 768  # Default dimension for text-embedding-004
        
        text = text.strip()[:20000]  # Truncate to max length
        
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=f"models/{self.embedding_model}",
                    content=text,
                    task_type="retrieval_document"
                )
                
                embedding = result.get('embedding', [])
                if embedding:
                    logger.debug(f"✅ Generated embedding of dimension {len(embedding)}")
                    return embedding
                else:
                    raise ValueError("Empty embedding returned")
            
            except genai.types.StopCandidateException as e:
                logger.warning(f"API safety filter triggered: {e}")
                return [0.0] * 768
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    logger.warning(f"Embedding failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    # Return zero vector as fallback
                    return [0.0] * 768

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        vec1_arr = np.array(vec1, dtype=np.float32)
        vec2_arr = np.array(vec2, dtype=np.float32)
        
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)

        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1_arr, vec2_arr) / (norm1 * norm2))

    def retrieve_relevant_rules(
        self,
        company: str,
        bill_embedding: List[float],
        bill_facts: Dict[str, Any],
        top_k: int = 10,
        policy_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant rules using vector similarity.
        Better category filtering and fallback handling.
        """
        try:
            query_filter = {"company": company, "status": "active"}
            
            if policy_name:
                query_filter["policy_name"] = policy_name
            
            policy_docs = list(self.rules_collection.find(query_filter))
            
            if not policy_docs:
                logger.warning(f"No active policies found for company={company}, policy_name={policy_name or 'N/A'}")
                # Try inactive policies as fallback
                query_filter.pop("status", None)
                policy_docs = list(self.rules_collection.find(query_filter))
            
            if not policy_docs:
                logger.warning(f"No policies at all found for company={company}")
                return []

            all_rules = []
            for doc in policy_docs:
                rules = doc.get('rules_extracted', [])
                if rules:
                    all_rules.extend(rules)

            if not all_rules:
                logger.warning(f"No rules found in policy documents")
                return []

            logger.info(f"Collected {len(all_rules)} rules from {len(policy_docs)} policy document(s)")

            # Better category filtering
            bill_category = bill_facts.get('category', '')
            if bill_category:
                # First try exact category match + "Other"
                filtered_rules = [
                    rule for rule in all_rules
                    if rule.get('category') in [bill_category, 'Other']
                ]
                
                # If no matches, keep all rules (don't filter by category)
                if not filtered_rules:
                    logger.info(f"No rules found for category '{bill_category}', using all {len(all_rules)} rules")
                    filtered_rules = all_rules
                else:
                    logger.info(f"Filtered to {len(filtered_rules)} rules matching category '{bill_category}'")
                
                all_rules = filtered_rules
            
            # Score rules safely
            scored_rules = []
            for rule in all_rules:
                if 'embedding' in rule and rule['embedding']:
                    try:
                        similarity = self.cosine_similarity(bill_embedding, rule['embedding'])
                        if similarity >= -1.0 and similarity <= 1.0:  # Valid similarity range
                            rule['similarity_score'] = similarity
                            scored_rules.append(rule)
                    except Exception as e:
                        logger.warning(f"Skipping rule {rule.get('rule_id')} due to embedding error: {e}")

            if not scored_rules:
                logger.warning(f"No rules with valid embeddings, returning first {top_k} raw rules")
                return all_rules[:top_k]

            # Sort by similarity
            scored_rules.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Better high-severity boosting
            high_severity_rules = [r for r in scored_rules if r.get('severity', '').upper() == 'HIGH']
            top_rules = scored_rules[:top_k]
            seen_rule_ids = {rule.get('rule_id') for rule in top_rules if rule.get('rule_id')}

            # Add HIGH severity rules that didn't make top_k
            for hs_rule in high_severity_rules:
                if hs_rule.get('rule_id') not in seen_rule_ids and len(top_rules) < top_k * 1.5:
                    top_rules.append(hs_rule)
                    seen_rule_ids.add(hs_rule.get('rule_id'))

            # Proper deduplication
            seen_texts = {}
            unique_rules = []
            
            for rule in top_rules:
                rule_text = rule.get('raw_text', '').strip().lower()
                rule_id = rule.get('rule_id')
                
                if rule_text not in seen_texts:
                    seen_texts[rule_text] = rule_id
                    unique_rules.append(rule)
                else:
                    logger.debug(f"Skipping duplicate: {rule_id} (similar to {seen_texts[rule_text]})")
            
            logger.info(f"Returning {len(unique_rules)} unique rules after deduplication and severity boosting")
            return unique_rules

        except Exception as e:
            logger.error(f"Error retrieving rules: {e}", exc_info=True)
            return []

    def _format_all_rules(self, policy_rules: List[Dict[str, Any]]) -> str:
        """Format rules with clear structure"""
        lines = []
        for idx, rule in enumerate(policy_rules, 1):
            lines.append(f"\n=== RULE {idx} (ID: {rule.get('rule_id')}) ===")
            lines.append(f"Category: {rule.get('category', 'N/A')}")
            lines.append(f"Severity: {rule.get('severity', 'MEDIUM')}")
            lines.append(f"Text: {rule.get('raw_text', 'N/A')}")
            
            attrs = rule.get('attributes', {})
            if attrs:
                lines.append("Parameters:")
                
                if attrs.get('currency'):
                    lines.append(f"  - Currency: {attrs.get('currency')} (REQUIRED)")
                if attrs.get('max_amount') is not None:
                    lines.append(f"  - Max Amount: {attrs.get('max_amount')} {attrs.get('currency', 'INR')}")
                if attrs.get('min_amount') is not None:
                    lines.append(f"  - Min Amount: {attrs.get('min_amount')} {attrs.get('currency', 'INR')}")
                if attrs.get('time_limit_days') is not None:
                    lines.append(f"  - Time Limit: {attrs.get('time_limit_days')} days")
                if attrs.get('allowed_modes'):
                    lines.append(f"  - Allowed: {', '.join(attrs.get('allowed_modes'))}")
                if attrs.get('disallowed_modes'):
                    lines.append(f"  - Prohibited: {', '.join(attrs.get('disallowed_modes'))}")
                if attrs.get('conditions'):
                    lines.append(f"  - Conditions: {', '.join(attrs.get('conditions'))}")
                if attrs.get('max_quantity'):
                    lines.append(f"  - Max Quantity: {attrs.get('max_quantity')}")
        
        return "\n".join(lines)
    
    def _format_bill_details(self, bill_facts: Dict[str, Any]) -> str:
        """Format bill details for LLM prompt"""
        bill_meta = bill_facts.get('bill_meta', {})
        
        lines = []
        lines.append(f"--- Expense Bill Summary ---")
        lines.append(f"Transaction ID: {bill_meta.get('transaction_id', 'N/A')}")
        lines.append(f"Category: {bill_meta.get('category', 'N/A')}")
        lines.append(f"Amount: {bill_meta.get('amount', 'N/A')} {bill_meta.get('currency', 'INR')}")
        
        if bill_meta.get('original_currency') and bill_meta.get('original_currency') != bill_meta.get('currency'):
            lines.append(f"Original Amount: {bill_meta.get('original_amount', 'N/A')} {bill_meta.get('original_currency')}")
        
        if bill_meta.get('date'):
            lines.append(f"Bill Date: {bill_meta.get('date')}")
            
            if 'days_since_bill' in bill_facts and 'analysis_date' in bill_facts:
                lines.append(f"Bill Age: {bill_facts['days_since_bill']} days old (as of {bill_facts['analysis_date']})")
            else:
                lines.append(f"Bill Age: Not calculated")

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
        if bill_meta.get('quantity') is not None:
            lines.append(f"Quantity: {bill_meta.get('quantity')}")
        if bill_meta.get('has_receipt') is not None:
            lines.append(f"Receipt Attached: {bill_meta.get('has_receipt')}")
        
        if bill_facts.get('raw_text'):
            lines.append(f"\n--- Original Bill Text ---\n{bill_facts.get('raw_text', '')[:500]}")
        
        return "\n".join(lines)

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response"""
        text = text.strip()
        
        # Try markdown JSON block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Find [...]
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json.loads(text[start_idx : end_idx + 1])
                return text[start_idx : end_idx + 1]
            except json.JSONDecodeError:
                pass
        
        # Find {...}
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json.loads(text[start_idx : end_idx + 1])
                return text[start_idx : end_idx + 1]
            except json.JSONDecodeError:
                pass

        logger.warning("Could not extract JSON from LLM response")
        return None

    def _validate_llm_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM result"""
        validated = {
            "rule_id": str(result.get("rule_id", "N/A")),
            "compliant": bool(result.get("compliant", False)),
            "classification": str(result.get("classification", "Other Violation")),
            "severity": str(result.get("severity", "MEDIUM")).upper(),
            "explanation": str(result.get("explanation", "No explanation provided.")),
            "confidence": float(result.get("confidence", 0.5)),
            "violation_details": result.get("violation_details", {})
        }
        
        if not isinstance(validated['violation_details'], dict):
            validated['violation_details'] = {}
        
        validated['violation_details']['expected'] = str(validated['violation_details'].get('expected', 'N/A'))
        validated['violation_details']['actual'] = str(validated['violation_details'].get('actual', 'N/A'))
        validated['violation_details']['deviation'] = str(validated['violation_details'].get('deviation', 'N/A'))
        
        validated['confidence'] = max(0.0, min(1.0, validated['confidence']))

        if validated['severity'] not in ['HIGH', 'MEDIUM', 'LOW']:
            validated['severity'] = 'MEDIUM'
        
        return validated

    def reason_with_llm_batch(
        self,
        bill_facts: Dict[str, Any],
        policy_rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check ALL rules in a SINGLE LLM call"""
        
        if not policy_rules:
            logger.info("No rules provided for batch reasoning")
            return []
        
        # Pre-calculate bill age
        bill_date_str = bill_facts.get('bill_meta', {}).get('date')
        if bill_date_str:
            try:
                bill_date_obj = datetime.fromisoformat(bill_date_str.replace('Z', '+00:00'))
                if bill_date_obj.tzinfo is None:
                    bill_date_obj = bill_date_obj.replace(tzinfo=timezone.utc)

                current_date = datetime.now(timezone.utc)
                days_old = (current_date - bill_date_obj).days
                
                bill_facts['days_since_bill'] = max(0, days_old)  # Never negative
                bill_facts['analysis_date'] = current_date.strftime('%Y-%m-%d')
                logger.info(f"Bill age: {days_old} days")
            except Exception as e:
                logger.warning(f"Could not calculate bill age: {e}")
                bill_facts.pop('days_since_bill', None)
                bill_facts.pop('analysis_date', None)

        clean_rules = []
        for rule in policy_rules:
            clean_rule = {
                'rule_id': rule.get('rule_id'),
                'category': rule.get('category'),
                'severity': rule.get('severity'),
                'raw_text': rule.get('raw_text'),
                'attributes': rule.get('attributes'),
                'applies_to': rule.get('applies_to'),
            }
            clean_rules.append(clean_rule)
        bill_description = bill_facts
        rules_text = clean_rules
        
        prompt = f"""
You are an expert compliance auditor. Your task is to check whether the following expense bill violates any of the listed policy rules.

Instructions:
- Carefully read each rule and the bill details.
- For each rule, decide if the bill violates it.
- ONLY include rules that are **violated** in your output.
- If the bill fully complies with a rule, do NOT include it in the result.
- Always base reasoning strictly on the rule text and bill details — no assumptions.

Output format:
Return ONLY a valid JSON array (no markdown, no prose) containing ONE object per violated rule.
If there are **no violations**, return an **empty array**: []

Each JSON object must have the following keys:
[
  {{
    "rule_id": "r1",
    "compliant": false,
    "classification": "Short descriptive phrase (e.g., 'Late Submission', 'Exceeds Limit', 'Currency Mismatch', 'Receipt Missing')",
    "severity": "HIGH|MEDIUM|LOW",
    "explanation": "1–2 sentence explanation referencing the rule text and bill data, explaining why it violates the rule.",
    "confidence": number between 0.0 and 1.0,
    "violation_details": {{
      "expected": "What the rule expects (e.g., 'Submission within 10 days', 'Currency must be INR', 'Receipt required')",
      "actual": "What the bill actually shows (e.g., 'Submitted after 30 days', 'Currency: AED', 'No receipt found')",
      "deviation": "Summarize the difference (e.g., '20 days late', 'Currency mismatch', 'Missing receipt')"
    }}
  }},
  ...
]

BILL DETAILS:
{bill_description}

POLICY RULES ({len(policy_rules)} total):
{rules_text}

Important:
- Return ONLY the violated rules — exclude compliant ones.
- Do NOT include markdown, comments, or explanations outside JSON.
- Ensure JSON is valid and can be parsed directly.
"""

        try:
            model = genai.GenerativeModel(self.model_name)
            logger.info(f"Sending batch reasoning prompt to LLM for {len(rules_text)} rules")
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=0.8,
                    max_output_tokens=8192,
                )
            )
            
            result_text = response.text.strip()
            logger.debug(f"LLM response (first 300 chars): {result_text[:300]}")
            
            result_json = self._extract_json(result_text)
            
            if not result_json:
                raise ValueError("Could not extract JSON from response")
            
            results = json.loads(result_json)
            
            if not isinstance(results, list):
                logger.warning(f"Expected list, got {type(results).__name__}")
                results = [results] if isinstance(results, dict) else []
            
            # Validate results
            enriched_results = []
            rule_map = {r.get('rule_id'): r for r in policy_rules if r.get('rule_id')}
            
            for result in results:
                result = self._validate_llm_result(result)
                
                rule_id = result.get('rule_id', '')
                if rule_id in rule_map:
                    rule = rule_map[rule_id]
                    result['company_rule_text'] = rule.get('raw_text', '')
                    result['rule_category'] = rule.get('category', '')
                
                result['bill_snippet'] = bill_facts.get('raw_text', '')[:200]
                result['model_used'] = self.model_name
                
                enriched_results.append(result)
            
            logger.info(f"Batch analysis complete: {len(enriched_results)} violations found")
            return enriched_results

        except json.JSONDecodeError as e:
            logger.error(f"JSON error: {e}")
            return self._fallback_batch_results(bill_facts, policy_rules, f"JSON parsing: {str(e)}")
        
        except Exception as e:
            logger.error(f"LLM batch error: {e}", exc_info=True)
            return self._fallback_batch_results(bill_facts, policy_rules, f"Analysis error: {str(e)}")

    def _fallback_batch_results(
        self,
        bill_facts: Dict[str, Any],
        policy_rules: List[Dict[str, Any]],
        error_msg: str
    ) -> List[Dict[str, Any]]:
        """Fallback when batch processing fails"""
        results = []
        for rule in policy_rules:
            results.append({
                "rule_id": rule.get('rule_id', 'unknown'),
                "compliant": False,
                "classification": "Analysis Error",
                "severity": rule.get('severity', 'MEDIUM'),
                "explanation": f"System error: {error_msg}",
                "confidence": 0.0,
                "violation_details": {"expected": "N/A", "actual": "N/A", "deviation": "System error"},
                "company_rule_text": rule.get('raw_text', ''),
                "rule_category": rule.get('category', ''),
                "bill_snippet": bill_facts.get('raw_text', '')[:200],
                "model_used": self.model_name,
                "error": error_msg
            })
        return results

    def reason_with_llm(
        self,
        bill_facts: Dict[str, Any],
        policy_rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        results = self.reason_with_llm_batch(bill_facts, [policy_rule])
        return results[0] if results else {
            "rule_id": policy_rule.get('rule_id'),
            "compliant": False,
            "classification": "Error",
            "severity": "MEDIUM",
            "explanation": "Batch processing returned no results",
            "confidence": 0.0
        }