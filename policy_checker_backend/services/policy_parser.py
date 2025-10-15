# services/improved_policy_parser.py - Enhanced policy extraction with better categorization

import google.generativeai as genai
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class PolicyParser:
    """
    Enhanced policy parser with:
    - Better category extraction and normalization
    - Comprehensive rule attribute extraction
    - Multiple validation layers
    """
    
    # Standard category taxonomy
    STANDARD_CATEGORIES = {
        "Travel": ["travel", "journey", "trip", "transport", "commute", "fare", "ticket"],
        "Accommodation": ["hotel", "accommodation", "lodging", "stay", "room", "boarding"],
        "Food": ["food", "meal", "lunch", "dinner", "breakfast", "refreshment", "beverage", "catering"],
        "Communication": ["phone", "mobile", "internet", "data", "call", "telecom", "communication"],
        "Medical": ["medical", "health", "medicine", "doctor", "hospital", "clinic", "treatment"],
        "Entertainment": ["entertainment", "client", "guest", "hospitality", "recreation"],
        "Supplies": ["supplies", "stationery", "equipment", "materials", "office"],
        "Training": ["training", "course", "education", "workshop", "seminar", "conference"],
        "Other": []  # Catch-all
    }
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Improved Policy Parser initialized")
    
    def parse_policy(self, policy_text: str, company: str) -> Dict[str, Any]:
        """
        Extract all rules and return structured data including categories.
        Returns: {
            "rules": List[Dict],
            "categories": List[str],
            "company": str,
            "extracted_at": str
        }
        """
        
        prompt = f"""You are an expert policy analyzer. Extract ALL compliance rules from this company policy document.

IMPORTANT INSTRUCTIONS:
1. Extract EVERY single rule, limit, condition, and restriction
2. Assign each rule to ONE primary category from this list: {list(self.STANDARD_CATEGORIES.keys())}
3. Be exhaustive - don't miss any rules
4. If a rule spans multiple categories, create separate rule entries for each category
5. Extract ALL attributes: amounts, modes, conditions, restrictions

CATEGORY DEFINITIONS:
- Travel: Transportation, fares, tickets, commute, journey
- Accommodation: Hotels, lodging, stays, rooms
- Food: Meals, refreshments, beverages, catering
- Communication: Phone, internet, calls, data
- Medical: Healthcare, medicine, treatment
- Entertainment: Client entertainment, hospitality
- Supplies: Office supplies, equipment, materials
- Training: Courses, workshops, conferences, education
- Other: Anything that doesn't fit above categories

For each rule, create a JSON object with:
{{
  "rule_id": "unique_id (r1, r2, etc.)",
  "category": "ONE of the standard categories above",
  "subcategory": "optional specific type (e.g., 'domestic_travel', 'hotel_3star')",
  "attributes": {{
    "max_amount": numeric_limit or null,
    "min_amount": numeric_limit or null,
    "currency": "INR|USD|EUR (default INR)",
    "scope": "per_trip|per_day|per_month|per_year|total|per_person",
    "allowed_modes": ["list", "of", "allowed", "options"] or null,
    "disallowed_modes": ["list", "of", "disallowed", "options"] or null,
    "conditions": ["prior_approval", "receipt_required", "manager_approval", etc.],
    "domestic_only": true|false|null,
    "international_only": true|false|null,
    "grade_restrictions": ["grade_levels", "if", "applicable"] or null,
    "time_restrictions": "description of time limits" or null,
    "quantity_limits": "description of quantity restrictions" or null,
    "vendor_restrictions": ["approved", "vendors"] or null,
    "any_other_restriction": "capture ANY other condition mentioned"
  }},
  "raw_text": "exact original policy text",
  "severity": "HIGH|MEDIUM|LOW",
  "applies_to": "all_employees|specific_grades|specific_departments|etc"
}}

POLICY DOCUMENT:
{policy_text}

Return ONLY a valid JSON object with this structure:
{{
  "categories_found": ["list of all categories that have rules"],
  "rules": [array of rule objects as defined above]
}}

Be exhaustive. Extract EVERY rule, even minor ones.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.9,
                )
            )
            result_text = response.text.strip()
            
            # Extract JSON
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            parsed_data = json.loads(result_text)
            
            # Validate and enrich
            validated_rules = []
            categories_set = set()
            
            rules = parsed_data.get('rules', [])
            for idx, rule in enumerate(rules):
                # Ensure required fields
                if not rule.get('rule_id'):
                    rule['rule_id'] = f"r{idx+1}"
                
                # Normalize category
                category = rule.get('category', 'Other')
                normalized_category = self._normalize_category(category)
                rule['category'] = normalized_category
                categories_set.add(normalized_category)
                
                # Ensure attributes exist
                if not rule.get('attributes'):
                    rule['attributes'] = {}
                
                # Set defaults
                if not rule.get('severity'):
                    rule['severity'] = self._infer_severity(rule)
                
                if not rule.get('applies_to'):
                    rule['applies_to'] = 'all_employees'
                
                # Add metadata
                rule['company'] = company
                rule['extracted_at'] = datetime.utcnow().isoformat()
                
                validated_rules.append(rule)
            
            result = {
                "rules": validated_rules,
                "categories": sorted(list(categories_set)),
                "company": company,
                "extracted_at": datetime.utcnow().isoformat(),
                "total_rules": len(validated_rules)
            }
            
            logger.info(f"Extracted {len(validated_rules)} rules across {len(categories_set)} categories")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {result_text[:500]}")
            # Fallback
            return self._fallback_extraction(policy_text, company)
        
        except Exception as e:
            logger.error(f"Error parsing policy: {e}")
            return self._fallback_extraction(policy_text, company)
    
    def _normalize_category(self, category: str) -> str:
        """Normalize category to standard taxonomy."""
        if not category:
            return "Other"
        
        category_lower = category.lower().strip()
        
        # Direct match
        for std_cat in self.STANDARD_CATEGORIES.keys():
            if category_lower == std_cat.lower():
                return std_cat
        
        # Fuzzy match using keywords
        for std_cat, keywords in self.STANDARD_CATEGORIES.items():
            if any(kw in category_lower for kw in keywords):
                return std_cat
        
        return "Other"
    
    def _infer_severity(self, rule: Dict[str, Any]) -> str:
        """Infer severity from rule attributes."""
        attrs = rule.get('attributes', {})
        
        # High severity if:
        # - Has amount limits
        # - Has disallowed modes
        # - Requires approval
        if attrs.get('max_amount') or attrs.get('disallowed_modes'):
            return "HIGH"
        
        conditions = attrs.get('conditions', [])
        if any(c in str(conditions).lower() for c in ['approval', 'manager', 'authorization']):
            return "HIGH"
        
        # Medium severity if has allowed modes or restrictions
        if attrs.get('allowed_modes') or attrs.get('domestic_only'):
            return "MEDIUM"
        
        return "LOW"
    
    def _fallback_extraction(self, policy_text: str, company: str) -> Dict[str, Any]:
        """Fallback extraction using rule-based approach."""
        import re
        
        rules = []
        categories_set = set()
        
        # Split into meaningful chunks
        sentences = re.split(r'[.!?]\s+', policy_text)
        
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Detect category
            category = self._detect_category_keywords(sentence)
            categories_set.add(category)
            
            # Detect amounts
            amount_match = re.search(r'(?:INR|Rs\.?|₹)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', sentence)
            max_amount = None
            if amount_match:
                max_amount = float(amount_match.group(1).replace(',', ''))
            
            # Basic rule structure
            rule = {
                "rule_id": f"r{idx+1}",
                "category": category,
                "attributes": {
                    "max_amount": max_amount,
                    "currency": "INR"
                },
                "raw_text": sentence,
                "severity": "MEDIUM",
                "company": company,
                "extracted_at": datetime.utcnow().isoformat()
            }
            
            rules.append(rule)
        
        logger.warning(f"Used fallback extraction: {len(rules)} rules")
        return {
            "rules": rules,
            "categories": sorted(list(categories_set)),
            "company": company,
            "extracted_at": datetime.utcnow().isoformat(),
            "total_rules": len(rules)
        }
    
    def _detect_category_keywords(self, text: str) -> str:
        """Detect category from text using keyword matching."""
        text_lower = text.lower()
        
        for category, keywords in self.STANDARD_CATEGORIES.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "Other"
    
    def get_category_list(self, company: str) -> List[str]:
        """Retrieve all categories for a company from database."""
        # This should query your MongoDB to get stored categories
        # For now, return standard categories
        return list(self.STANDARD_CATEGORIES.keys())