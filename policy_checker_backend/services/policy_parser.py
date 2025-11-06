# services/optimized_policy_parser_fixed.py - Properly optimized with maintained quality

import google.generativeai as genai
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import re

load_dotenv()
logger = logging.getLogger(__name__)

class PolicyParser:
    """
    ✅ PROPERLY OPTIMIZED Policy parser:
    - Maintains comprehensive extraction (22+ rules)
    - Improved prompt with explicit instructions
    - Better retry logic based on content analysis
    - Enhanced fallback mechanisms
    """
    
    STANDARD_CATEGORIES = {
        "Travel": ["travel", "journey", "trip", "transport", "commute", "fare", "ticket"],
        "Accommodation": ["hotel", "accommodation", "lodging", "stay", "room", "boarding"],
        "Food": ["food", "meal", "lunch", "dinner", "breakfast", "refreshment", "beverage", "catering"],
        "Communication": ["phone", "mobile", "internet", "data", "call", "telecom", "communication"],
        "Medical": ["medical", "health", "medicine", "doctor", "hospital", "clinic", "treatment"],
        "Entertainment": ["entertainment", "client", "guest", "hospitality", "recreation"],
        "Supplies": ["supplies", "stationery", "equipment", "materials", "office"],
        "Training": ["training", "course", "education", "workshop", "seminar", "conference"],
        "Other": []
    }
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("✅ Properly Optimized Policy Parser initialized")
    
    def parse_policy(self, policy_text: str, company: str) -> Dict[str, Any]:
        """
        Enhanced parsing with maintained quality:
        - Comprehensive prompt (based on old working version)
        - Smart retry logic
        - Better JSON extraction
        """
        
        # MAIN PROMPT - Based on your original working version but slightly optimized
        main_prompt = f"""You are an expert policy analyzer. Extract ALL compliance rules from this company policy document.

CRITICAL INSTRUCTIONS:
1. Extract EVERY single rule, limit, condition, and restriction - be EXHAUSTIVE
2. If one sentence contains multiple rules, create separate entries for each
3. If a rule applies to multiple categories, duplicate it under each category
4. Assign each rule to ONE primary category from: {list(self.STANDARD_CATEGORIES.keys())}
5. Extract ALL attributes comprehensively - don't leave nulls if info exists

CATEGORY DEFINITIONS (choose the BEST fit):
- Travel: Transportation, fares, tickets, commute, journey, cab, bus, train, flight
- Accommodation: Hotels, lodging, stays, rooms, boarding, guest house
- Food: Meals, refreshments, beverages, catering, lunch, dinner, breakfast, snacks
- Communication: Phone, mobile, internet, data, calls, telecom, broadband
- Medical: Healthcare, medicine, treatment, doctor, hospital, clinic, insurance
- Entertainment: Client entertainment, hospitality, recreation, events
- Supplies: Office supplies, equipment, materials, stationery
- Training: Courses, workshops, conferences, seminars, education, learning
- Other: Anything that doesn't clearly fit above categories

RULE STRUCTURE - Each rule MUST have:
{{
  "rule_id": "unique_id (r1, r2, r3...)",
  "category": "ONE of the standard categories",
  "subcategory": "specific type (e.g., 'domestic_flight', 'hotel_booking', '3star_hotel')",
  "attributes": {{
    "max_amount": numeric_value or null,
    "min_amount": numeric_value or null,
    "currency": "INR|USD|EUR" (default INR),
    "scope": "per_trip|per_day|per_month|per_year|total|per_person|per_night",
    "allowed_modes": ["list", "of", "allowed", "options"] or null,
    "disallowed_modes": ["list", "of", "disallowed", "items"] or null,
    "conditions": ["prior_approval", "receipt_required", "manager_approval", "HR_approval", etc.],
    "domestic_only": true|false|null,
    "international_only": true|false|null,
    "grade_restrictions": ["grade_levels"] or null,
    "time_restrictions": "description" or null,
    "quantity_limits": "description" or null,
    "vendor_restrictions": ["approved_vendors"] or null,
    "any_other_restriction": "capture ANY other limitation mentioned"
  }},
  "raw_text": "exact original policy text from document",
  "severity": "HIGH|MEDIUM|LOW",
  "applies_to": "all_employees|specific_grades|specific_departments|etc"
}}

EXAMPLES:
Policy: "Employees traveling domestically can book economy flights up to INR 15,000. Business class requires VP approval."
Rules:
[
  {{
    "rule_id": "r1",
    "category": "Travel",
    "subcategory": "domestic_flight",
    "attributes": {{
      "max_amount": 15000,
      "currency": "INR",
      "scope": "per_trip",
      "allowed_modes": ["economy"],
      "conditions": [],
      "domestic_only": true
    }},
    "raw_text": "Employees traveling domestically can book economy flights up to INR 15,000.",
    "severity": "HIGH"
  }},
  {{
    "rule_id": "r2",
    "category": "Travel",
    "subcategory": "domestic_flight_business_class",
    "attributes": {{
      "allowed_modes": ["business_class"],
      "conditions": ["VP_approval"],
      "domestic_only": true
    }},
    "raw_text": "Business class requires VP approval.",
    "severity": "HIGH"
  }}
]

Policy: "Hotel stays limited to 3-star properties, max INR 5,000/night. Alcohol not reimbursable."
Rules:
[
  {{
    "rule_id": "r3",
    "category": "Accommodation",
    "subcategory": "hotel_3star",
    "attributes": {{
      "max_amount": 5000,
      "currency": "INR",
      "scope": "per_night",
      "quantity_limits": "3-star properties only",
      "conditions": ["receipt_required"]
    }},
    "raw_text": "Hotel stays limited to 3-star properties, max INR 5,000/night.",
    "severity": "HIGH"
  }},
  {{
    "rule_id": "r4",
    "category": "Food",
    "subcategory": "beverages",
    "attributes": {{
      "disallowed_modes": ["alcohol", "alcoholic_beverages"],
      "any_other_restriction": "no reimbursement for alcohol"
    }},
    "raw_text": "Alcohol not reimbursable.",
    "severity": "HIGH"
  }}
]

POLICY DOCUMENT TO ANALYZE:
{policy_text}

RETURN FORMAT - Valid JSON only:
{{
  "categories_found": ["list", "of", "all", "categories"],
  "rules": [array of rule objects following structure above]
}}

REMEMBER: Be EXHAUSTIVE. Extract EVERY rule, even minor ones. Don't summarize - create individual entries.
"""

        def extract_json_robust(text: str) -> str:
            """More robust JSON extraction"""
            text = text.strip()
            
            # Try markdown code blocks first
            if '```json' in text:
                parts = text.split('```json', 1)
                if len(parts) > 1:
                    json_part = parts[1].split('```', 1)[0].strip()
                    return json_part
            
            if '```' in text:
                parts = text.split('```', 1)
                if len(parts) > 1:
                    json_part = parts[1].split('```', 1)[0].strip()
                    return json_part
            
            # Find first { to last }
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                return text[first_brace:last_brace + 1]
            
            return text
        
        try:
            # Primary extraction attempt
            logger.info("🔍 Starting comprehensive rule extraction...")
            response = self.model.generate_content(
                main_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temp for consistency
                    top_p=0.95,
                    max_output_tokens=16384,  # Increased for comprehensive extraction
                )
            )
            
            result_text = response.text.strip()
            json_text = extract_json_robust(result_text)
            parsed_data = json.loads(json_text)
            
            rules = parsed_data.get('rules', [])
            initial_count = len(rules)
            logger.info(f"📊 Initial extraction: {initial_count} rules")
            
            # Smart retry logic based on policy length
            policy_word_count = len(policy_text.split())
            expected_min_rules = max(15, policy_word_count // 100)  # Dynamic threshold
            
            if initial_count < expected_min_rules:
                logger.warning(f"⚠️ Only {initial_count} rules found, expected ~{expected_min_rules}. Retrying with emphasis...")
                
                retry_prompt = main_prompt + f"""

⚠️ IMPORTANT: The previous extraction only found {initial_count} rules, but this policy document has {policy_word_count} words and likely contains MORE rules.

PLEASE RE-ANALYZE and extract:
- Every numeric limit (amounts, quantities, time periods)
- Every approval requirement
- Every restriction or prohibition
- Every condition or qualification
- Every allowed/disallowed item or mode
- Every grade-specific rule
- Every subcategory variation

Be MORE GRANULAR. If a sentence mentions multiple limits or conditions, create SEPARATE rules for each.
"""
                
                response2 = self.model.generate_content(
                    retry_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.05,  # Even lower temp
                        top_p=0.9,
                        max_output_tokens=16384,
                    )
                )
                
                json_text2 = extract_json_robust(response2.text.strip())
                parsed_data2 = json.loads(json_text2)
                
                rules2 = parsed_data2.get('rules', [])
                logger.info(f"📊 Retry extraction: {len(rules2)} rules")
                
                # Use whichever result has more rules
                if len(rules2) > initial_count:
                    parsed_data = parsed_data2
                    logger.info(f"✅ Using retry result ({len(rules2)} rules)")
            
            # Validate and enrich (same as original)
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
            
            logger.info(f"✅ Final result: {len(validated_rules)} rules across {len(categories_set)} categories")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            logger.error(f"Raw response preview: {result_text[:500] if 'result_text' in locals() else 'N/A'}")
            return self._enhanced_fallback_extraction(policy_text, company)
        
        except Exception as e:
            logger.error(f"❌ Error parsing policy: {e}")
            return self._enhanced_fallback_extraction(policy_text, company)
    
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
        if attrs.get('max_amount') or attrs.get('disallowed_modes'):
            return "HIGH"
        
        conditions = attrs.get('conditions', [])
        if any(c in str(conditions).lower() for c in ['approval', 'manager', 'authorization']):
            return "HIGH"
        
        # Medium severity if has restrictions
        if attrs.get('allowed_modes') or attrs.get('domestic_only') or attrs.get('grade_restrictions'):
            return "MEDIUM"
        
        return "LOW"
    
    def _enhanced_fallback_extraction(self, policy_text: str, company: str) -> Dict[str, Any]:
        """Enhanced fallback with better rule detection"""
        logger.warning("⚠️ Using enhanced fallback extraction...")
        
        rules = []
        categories_set = set()
        
        # Multiple splitting strategies
        segments = []
        
        # Strategy 1: Split by bullets and numbered lists
        bullet_pattern = r'[\n\r]+\s*[\u2022\u2023\u25E6\-\*•]\s*|\n\s*\d+[\.)]\s*'
        for part in re.split(bullet_pattern, policy_text):
            part = part.strip()
            if len(part) > 20:
                segments.append(part)
        
        # Strategy 2: Split remaining by sentence boundaries
        expanded_segments = []
        for seg in segments:
            # Split on period followed by capital letter or number
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', seg)
            expanded_segments.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        
        # Strategy 3: Split on semicolons (often separate rules)
        final_segments = []
        for seg in expanded_segments:
            if ';' in seg:
                final_segments.extend([s.strip() for s in seg.split(';') if len(s.strip()) > 15])
            else:
                final_segments.append(seg)
        
        # Enhanced patterns
        amount_pattern = re.compile(r'(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.\d+)?)', re.IGNORECASE)
        per_pattern = re.compile(r'per\s+(day|month|trip|person|year|night|visit|journey)', re.IGNORECASE)
        max_pattern = re.compile(r'(?:up\s+to|maximum|max|limit|not\s+exceed)', re.IGNORECASE)
        
        for idx, text in enumerate(final_segments):
            text_lower = text.lower()
            
            # Detect category
            category = self._detect_category_keywords(text)
            categories_set.add(category)
            
            # Extract amount
            max_amount = None
            amount_match = amount_pattern.search(text)
            if amount_match:
                try:
                    max_amount = float(amount_match.group(1).replace(',', ''))
                except:
                    pass
            
            # Extract scope
            scope = None
            per_match = per_pattern.search(text)
            if per_match:
                scope = f"per_{per_match.group(1).lower()}"
            
            # Detect conditions
            conditions = []
            if 'receipt' in text_lower:
                conditions.append('receipt_required')
            if 'approval' in text_lower:
                if 'prior' in text_lower or 'advance' in text_lower:
                    conditions.append('prior_approval')
                if 'manager' in text_lower:
                    conditions.append('manager_approval')
                if 'hr' in text_lower:
                    conditions.append('HR_approval')
            
            # Detect restrictions
            disallowed = []
            negative_phrases = ['not reimbursable', 'not allowed', 'prohibited', 
                              'not permitted', 'excluded', 'disallowed', 'not covered']
            for phrase in negative_phrases:
                if phrase in text_lower:
                    # Extract what's being disallowed
                    words = text.split()
                    for i, word in enumerate(words):
                        if phrase.split()[0] in word.lower():
                            if i > 0:
                                disallowed.append(words[i-1].lower())
            
            # Determine severity
            severity = "HIGH" if (max_amount or disallowed or 'approval' in text_lower) else "MEDIUM"
            
            rule = {
                "rule_id": f"r{idx+1}",
                "category": category,
                "attributes": {
                    "max_amount": max_amount,
                    "currency": "INR" if max_amount else None,
                    "scope": scope,
                    "disallowed_modes": disallowed if disallowed else None,
                    "conditions": conditions if conditions else None,
                },
                "raw_text": text,
                "severity": severity,
                "applies_to": "all_employees",
                "company": company,
                "extracted_at": datetime.utcnow().isoformat()
            }
            
            rules.append(rule)
        
        logger.info(f"📊 Fallback extracted {len(rules)} rules")
        
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
        
        # Try each category's keywords
        for category, keywords in self.STANDARD_CATEGORIES.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "Other"
    
    def get_category_list(self, company: str) -> List[str]:
        """Get standard categories."""
        return list(self.STANDARD_CATEGORIES.keys())