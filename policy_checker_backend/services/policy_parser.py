# services/improved_policy_parser.py - Enhanced with better rule categorization

import google.generativeai as genai
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
import re
import hashlib

load_dotenv()
logger = logging.getLogger(__name__)

class PolicyParser:
    """
    Improved Policy Parser with:
    - Better rule categorization and attributes extraction
    - Proper rule deduplication
    - Consistent extraction
    - Better validation
    """
    
    STANDARD_CATEGORIES = {
        "Travel": ["travel", "journey", "trip", "transport", "commute", "fare", "ticket", "flight", "train", "cab", "bus", "uber", "ola"],
        "Accommodation": ["hotel", "accommodation", "lodging", "stay", "room", "boarding"],
        "Food": ["food", "meal", "lunch", "dinner", "breakfast", "refreshment", "beverage", "catering", "coffee", "tea"],
        "Communication": ["phone", "mobile", "internet", "data", "call", "telecom", "communication"],
        "Medical": ["medical", "health", "medicine", "doctor", "hospital", "clinic", "treatment"],
        "Entertainment": ["entertainment", "client", "guest", "hospitality", "recreation"],
        "Supplies": ["supplies", "stationery", "equipment", "materials", "office", "software", "tools"],
        "Training": ["training", "course", "education", "workshop", "seminar", "conference"],
        "Other": ["other", "miscellaneous", "general"]
    }
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("✅ Improved Policy Parser initialized")
    
    def _generate_rule_hash(self, rule_text: str, category: str, attributes: Dict[str, Any]) -> str:
        """Generate a unique hash for a rule to detect duplicates"""
        normalized_text = ' '.join(rule_text.lower().split())
        
        # FIX: Handle case where key exists but value is explicitly None (JSON null)
        # using 'or []' ensures that if .get() returns None, we treat it as an empty list.
        key_attrs = {
            'max_amount': attributes.get('max_amount'),
            'min_amount': attributes.get('min_amount'),
            'allowed_modes': sorted(attributes.get('allowed_modes') or []),
            'disallowed_modes': sorted(attributes.get('disallowed_modes') or []),
            'conditions': sorted(attributes.get('conditions') or [])
        }
        
        # Use default=str in json.dumps to handle any unexpected types cleanly
        hash_input = f"{normalized_text}|{category}|{json.dumps(key_attrs, sort_keys=True, default=str)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _deduplicate_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate rules based on content similarity."""
        seen_hashes = {}
        unique_rules = []
        
        for rule in rules:
            rule_hash = self._generate_rule_hash(
                rule.get('raw_text', ''),
                rule.get('category', ''),
                rule.get('attributes', {})
            )
            
            if rule_hash not in seen_hashes:
                seen_hashes[rule_hash] = rule
                unique_rules.append(rule)
            else:
                existing = seen_hashes[rule_hash]
                existing_attrs = len([v for v in existing.get('attributes', {}).values() if v])
                new_attrs = len([v for v in rule.get('attributes', {}).values() if v])
                
                if new_attrs > existing_attrs:
                    unique_rules.remove(existing)
                    unique_rules.append(rule)
                    seen_hashes[rule_hash] = rule
                    logger.info(f"Replaced duplicate rule: {rule.get('rule_id')}")
        
        logger.info(f"Deduplication: {len(rules)} → {len(unique_rules)} rules")
        return unique_rules
    
    def parse_policy(self, policy_text: str, company: str) -> Dict[str, Any]:
        """
        OPTIMIZED: Forces extraction of 'Eligible Expenses' and 'Procedural' rules.
        """
        
        main_prompt = f"""
You are an expert Policy Analyst. Extract a comprehensive list of compliance rules from the text below.

TARGET OUTPUT: A JSON object containing a list of rule objects.

CRITICAL INSTRUCTIONS:
1. **EXTRACT EVERYTHING**:
   - **Eligible Expenses:** Treat every item in an "Eligible" or "Allowed" list as a separate POSITIVE rule (e.g., "Local transportation is allowed").
   - **Prohibited Expenses:** Treat every item in a "Non-Reimbursable" list as a separate NEGATIVE rule.
   - **Procedural Rules:** Extract rules about process (e.g., "Must attach scanned photos", "Must complete Request Form").
   - **Limits:** Extract currency, amounts, and time limits.

2. **RULE STRUCTURE**:
   - If the text says "Local transportation (Uber/Ola)", create a rule: {{ "raw_text": "Local transportation (e.g., Uber/Ola, auto, fuel) is eligible", "rule_type": "positive", "category": "Travel" }}
   - If the text says "Alcoholic beverages", create a rule: {{ "raw_text": "Alcoholic beverages are non-reimbursable", "rule_type": "negative", "category": "Food" }}

3. **ATTRIBUTES**:
   - If a rule implies a currency (e.g., "₹100"), set "currency": "AED".
   - Use 'allowed_modes' for specific examples given in positive rules.

STANDARD CATEGORIES:
Travel, Accommodation, Food, Communication, Medical, Entertainment, Supplies, Training, Other

INPUT POLICY:
{policy_text}

OUTPUT FORMAT (JSON):
{{
  "rules": [
    {{
      "rule_id": "string",
      "category": "string",
      "is_global": boolean,
      "rule_type": "positive" | "negative",
      "raw_text": "string (Full sentence describing the rule)",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "attributes": {{
        "max_amount": number | null,
        "min_amount": number | null,
        "currency": "string" | null,
        "submission_limit_days": number | null,
        "allowed_modes": ["string"] | null,
        "disallowed_modes": ["string"] | null,
        "receipt_required": boolean | null,
        "approval_required": boolean | null
      }}
    }}
  ]
}}
"""

        try:
            logger.info("🔍 Starting detailed rule extraction...")
            
            response = self.model.generate_content(
                main_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1, # Slightly higher to capture variety
                    response_mime_type="application/json",
                    max_output_tokens=16384,
                )
            )
            
            parsed_data = json.loads(response.text)
            rules = parsed_data.get('rules', [])
            
            logger.info(f"📊 Initial extraction: {len(rules)} rules")
            
            # Deduplicate
            rules = self._deduplicate_rules(rules)
            
            validated_rules = []
            categories_set = set()
            
            for idx, rule in enumerate(rules):
                rule['rule_id'] = f"r{idx+1}"
                
                # Normalize Category
                cat = self._normalize_category(rule.get('category'))
                rule['category'] = cat
                categories_set.add(cat)
                
                if not rule.get('attributes'):
                    rule['attributes'] = {}
                
                # Cleanup raw_text for negative rules if they are just keywords
                # Example: If text is just "Alcohol", change to "Alcohol is prohibited"
                if rule.get('rule_type') == 'negative' and len(rule.get('raw_text', '').split()) < 3:
                     rule['raw_text'] = f"{rule['raw_text']} is prohibited/non-reimbursable"

                # Generate Search Text
                attrs_str = ", ".join([f"{k}:{v}" for k, v in rule['attributes'].items() if v])
                rule['search_text'] = (
                    f"Category: {cat} | "
                    f"Type: {rule.get('rule_type', 'positive')} | "
                    f"Rule: {rule.get('raw_text')} | "
                    f"Limits: {attrs_str}"
                )
                
                if not rule.get('severity'):
                    rule['severity'] = self._infer_severity(rule)
                
                rule['company'] = company
                rule['extracted_at'] = datetime.now(timezone.utc).isoformat()
                
                validated_rules.append(rule)
            
            result = {
                "rules": validated_rules,
                "categories": sorted(list(categories_set)),
                "company": company,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "total_rules": len(validated_rules)
            }
            
            logger.info(f"✅ Final: {len(validated_rules)} unique rules extracted")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing policy: {e}", exc_info=True)
            return self._enhanced_fallback_extraction(policy_text, company)
        
    def _normalize_category(self, category: str) -> str:
        """Normalize category to standard taxonomy"""
        if not category:
            return "Other"
        
        category_lower = category.lower().strip()
        
        for std_cat in self.STANDARD_CATEGORIES.keys():
            if category_lower == std_cat.lower():
                return std_cat
        
        for std_cat, keywords in self.STANDARD_CATEGORIES.items():
            if any(kw in category_lower for kw in keywords):
                return std_cat
        
        return "Other"
    
    def _infer_severity(self, rule: Dict[str, Any]) -> str:
        """Infer severity from rule content"""
        attrs = rule.get('attributes', {})
        raw_text = rule.get('raw_text', '').lower()
        rule_type = rule.get('rule_type', 'positive')
        
        high_indicators = [
            attrs.get('max_amount') is not None,
            attrs.get('time_limit_days') is not None,
            attrs.get('receipt_required') == True,
            attrs.get('approval_required') == True,
            'mandatory' in raw_text,
            'must' in raw_text,
            'required' in raw_text,
            'prohibited' in raw_text,
            'not allowed' in raw_text,
            rule_type == 'negative'  # Negative rules (prohibitions) are high severity
        ]
        
        if any(high_indicators):
            return "HIGH"
        
        medium_indicators = [
            'should' in raw_text,
            'recommend' in raw_text,
            'may apply' in raw_text,
            'preferred' in raw_text,
        ]
        
        if any(medium_indicators):
            return "MEDIUM"
        
        return "LOW"
    
    def _enhanced_fallback_extraction(self, policy_text: str, company: str) -> Dict[str, Any]:
        """Fallback extraction"""
        logger.warning("⚠️ Using fallback extraction...")
        
        rules = []
        categories_set = {"Other"}
        
        segments = re.split(r'[\n\r]+\s*[\u2022\-\*•]\s*|\n\s*\d+[\.)]\s*', policy_text)
        
        for idx, text in enumerate(segments):
            text = text.strip()
            if len(text) < 20:
                continue
            
            rule = {
                "rule_id": f"r{idx+1}",
                "category": "Other",
                "rule_type": "negative" if any(word in text.lower() for word in ['not', 'prohibited', 'cannot', 'disallowed']) else "positive",
                "attributes": {},
                "raw_text": text,
                "severity": "HIGH" if any(word in text.lower() for word in ['must', 'required', 'mandatory', 'prohibited']) else "MEDIUM",
                "applies_to": "all_employees",
                "company": company,
                "extracted_at": datetime.now(timezone.utc).isoformat()
            }
            
            rules.append(rule)
        
        return {
            "rules": rules,
            "categories": sorted(list(categories_set)),
            "company": company,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "total_rules": len(rules)
        }
    
    def get_category_list(self, company: str) -> List[str]:
        """Get standard categories"""
        return list(self.STANDARD_CATEGORIES.keys())