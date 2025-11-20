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
        
        key_attrs = {
            'max_amount': attributes.get('max_amount'),
            'min_amount': attributes.get('min_amount'),
            'allowed_modes': sorted(attributes.get('allowed_modes', [])),
            'disallowed_modes': sorted(attributes.get('disallowed_modes', [])),
            'conditions': sorted(attributes.get('conditions', []))
        }
        
        hash_input = f"{normalized_text}|{category}|{json.dumps(key_attrs, sort_keys=True)}"
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
        """Enhanced parsing with better rule extraction"""
        
        main_prompt = f"""You are an expert HR policy analyzer. Extract detailed compliance rules from this policy document.

CRITICAL INSTRUCTIONS:
1. Extract EVERY distinct rule as a SEPARATE item
2. Differentiate between POSITIVE rules (what's allowed/required) and NEGATIVE rules (what's prohibited)
3. Each rule = ONE specific constraint or requirement
4. Assign each rule to appropriate category
5. Extract ALL attributes mentioned in the policy

STANDARD CATEGORIES:
Travel, Accommodation, Food, Communication, Medical, Entertainment, Supplies, Training, Other

RULE STRUCTURE:
{{
  "rule_id": "unique_id",
  "category": "one of the standard categories",
  "rule_type": "positive" (what's allowed/required) or "negative" (what's prohibited),
  "raw_text": "exact rule text from policy",
  "attributes": {{
    "max_amount": number or null,
    "min_amount": number or null,
    "currency": "INR|USD|EUR" (only if amount specified),
    "time_limit_days": number or null (e.g., "within 10 days" → 10),
    "receipt_required": true|false|null,
    "approval_required": true|false|null,
    "conditions": ["list", "of", "conditions"],
    "allowed_items": ["list"] or null,
    "disallowed_items": ["list"] or null,
    "notes": "any other important details"
  }},
  "severity": "HIGH" (mandatory/absolute rules) or "MEDIUM" (recommended/conditional),
  "applies_to": "all_employees"
}}

IMPORTANT RULES FOR THIS POLICY:
- "Receipts mandatory above ₹300" = Create ONE rule about receipt requirement with min_amount: 300
- "Submit within 10 working days" = Create ONE rule about time_limit_days: 10
- "All reimbursements in INR" = Create ONE rule about currency: "INR"
- "Pre-approval required" = Create ONE rule about approval_required: true
- "Personal purchases not reimbursable" = Create ONE rule with rule_type: "negative"
- "Daily limits may apply" = Create ONE rule noting limits apply (no specific amount)

POLICY DOCUMENT:
{policy_text}

Return ONLY valid JSON in format:
{{
  "categories_found": ["list of categories"],
  "rules": [array of rule objects with all details]
}}

IMPORTANT: 
- Extract each distinct rule separately
- Include both positive (allowed) and negative (prohibited) rules
- Be precise with amounts and time limits
- Keep raw_text as exact quote from policy
"""

        def extract_json_robust(text: str) -> str:
            """Robust JSON extraction"""
            text = text.strip()
            
            if '```json' in text:
                parts = text.split('```json', 1)
                if len(parts) > 1:
                    return parts[1].split('```', 1)[0].strip()
            
            if '```' in text:
                parts = text.split('```', 1)
                if len(parts) > 1:
                    return parts[1].split('```', 1)[0].strip()
            
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            
            if first_brace != -1 and last_brace != -1:
                return text[first_brace:last_brace + 1]
            
            return text
        
        try:
            logger.info("🔍 Starting rule extraction...")
            
            response = self.model.generate_content(
                main_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=0.9,
                    max_output_tokens=16384,
                )
            )
            
            result_text = response.text.strip()
            json_text = extract_json_robust(result_text)
            parsed_data = json.loads(json_text)
            
            rules = parsed_data.get('rules', [])
            logger.info(f"📊 Initial extraction: {len(rules)} rules")
            
            # Deduplicate
            rules = self._deduplicate_rules(rules)
            logger.info(f"📊 After deduplication: {len(rules)} unique rules")
            
            # Validate and enrich
            validated_rules = []
            categories_set = set()
            
            for idx, rule in enumerate(rules):
                if not rule.get('rule_id'):
                    rule['rule_id'] = f"r{idx+1}"
                else:
                    rule['rule_id'] = f"r{idx+1}"
                
                category = rule.get('category', 'Other')
                normalized_category = self._normalize_category(category)
                rule['category'] = normalized_category
                categories_set.add(normalized_category)
                
                if not rule.get('attributes'):
                    rule['attributes'] = {}
                
                attrs = rule['attributes']
                rule_text_lower = rule.get('raw_text', '').lower()
                
                # Only keep currency if amount is specified or currency is mentioned
                if attrs.get('currency'):
                    has_amount = attrs.get('max_amount') or attrs.get('min_amount')
                    mentions_currency = any(keyword in rule_text_lower for keyword in [
                        'currency', 'inr', 'usd', 'eur', 'aed', 'gbp', '₹', 'rs.', 
                        'processed in', 'reimbursed in', 'paid in'
                    ])
                    
                    if not (has_amount or mentions_currency):
                        attrs.pop('currency', None)
                
                if attrs.get('currency'):
                    attrs['currency'] = attrs['currency'].upper()
                
                if not rule.get('severity'):
                    rule['severity'] = self._infer_severity(rule)
                
                if not rule.get('applies_to'):
                    rule['applies_to'] = 'all_employees'
                
                if not rule.get('rule_type'):
                    rule['rule_type'] = 'positive'
                
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
            
            logger.info(f"✅ Final: {len(validated_rules)} unique rules across {len(categories_set)} categories")
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