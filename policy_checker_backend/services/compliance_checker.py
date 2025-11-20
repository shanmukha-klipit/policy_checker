# services/deterministic_compliance_checker.py - Generic approach (updated logic only)

from typing import List, Dict, Any, Tuple, Optional
import logging
from services.rag_engine import RAGEngine
from rapidfuzz import fuzz
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ComplianceChecker:
    """
    Enhanced compliance checker with:
    1. Generic deterministic checks (works for ANY rule attribute)
    2. LLM analysis for complex rules
    3. Consistent results across runs
    """
    
    CATEGORY_MATCH_THRESHOLD = 80
    
    def __init__(self):
        self.rag_engine = RAGEngine()
        logger.info("Deterministic Compliance Checker initialized")
    
    def check_compliance(
        self,
        bill_facts: Dict[str, Any],
        company: str,
        stored_categories: List[str],
        policy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced compliance check with deterministic pre-validation
        """
        
        mismatches = []
        bill_meta = bill_facts.get('bill_meta', {})
        bill_category = bill_facts.get("category", bill_meta.get("category", "")).strip()
        
        # Step 1: Category validation
        category_valid, matched_category, similarity = self._validate_category(
            bill_category, 
            stored_categories
        )
        
        if not category_valid:
            mismatches.append({
                "classification": "Unrecognized Category",
                "severity": "HIGH",
                "explanation": (f"Bill category '{bill_category}' does not match policy categories. "
                                f"Closest: '{matched_category}' ({similarity}%). "
                                f"Valid: {', '.join(stored_categories)}"),
                "confidence": 0.99,
                "company_rule_text": "Category validation",
                "bill_snippet": f"Category: {bill_category}",
                "suggested_category": matched_category,
                "rule_id": "CATEGORY_VALIDATION"
            })
        
        # Update with matched category
        if matched_category and similarity >= self.CATEGORY_MATCH_THRESHOLD:
            bill_facts['category'] = matched_category
            bill_meta['category'] = matched_category
        
        # Step 2: Retrieve relevant rules
        # bill_text =self._construct_bill_text(bill_facts)
        bill_text=bill_facts
        # bill_text = bill_facts.get('raw_text', '') or self._construct_bill_text(bill_facts)
        bill_embedding = self.rag_engine.generate_embedding(bill_text)
        
        relevant_rules = self.rag_engine.retrieve_relevant_rules(
            company=company,
            bill_embedding=bill_embedding,
            bill_facts=bill_facts,
            top_k=10,
            policy_name=policy_name
        )
        
        if not relevant_rules:
            logger.warning(f"No rules found for company={company}, category={bill_category}")
            mismatches.append({
                "classification": "No Policy Rules Found",
                "severity": "MEDIUM",
                "explanation": f"No policy rules found for {company}, category: {bill_category}",
                "confidence": 0.95,
                "company_rule_text": "N/A",
                "bill_snippet": bill_text[:200],
                "rule_id": "NO_RULES_FOUND"
            })
        
        # Step 3: GENERIC DETERMINISTIC PRE-CHECKS (works for ANY rule)
        # logger.info(f"Running generic deterministic pre-checks on {len(relevant_rules)} rules")
        # deterministic_violations = self._run_deterministic_checks(bill_facts, relevant_rules)
        # mismatches.extend(deterministic_violations)
        
        # Step 4: LLM analysis for rules that couldn't be checked deterministically
        # rules_for_llm = [r for r in relevant_rules if not self._is_deterministically_checkable(r)]
        
        rules_for_llm = [r for r in relevant_rules ]
        
        if rules_for_llm:
            logger.info(f"Running LLM analysis on {len(rules_for_llm)} complex rules")
            try:
                batch_results = self.rag_engine.reason_with_llm_batch(bill_facts, rules_for_llm)
                
                for result in batch_results:
                    if not result.get('compliant', True):
                        mismatches.append(result)
                        
            except Exception as e:
                logger.error(f"LLM batch analysis error: {e}", exc_info=True)
                mismatches.append({
                    "classification": "Analysis Error",
                    "severity": "HIGH",
                    "explanation": f"System error during analysis: {str(e)}",
                    "confidence": 0.1,
                    "company_rule_text": "System Error",
                    "bill_snippet": bill_text[:200],
                    "rule_id": "ANALYSIS_ERROR"
                })
        
        # Step 5: Deduplicate and sort
        deduplicated_mismatches = self._deduplicate_violations(mismatches)
        deduplicated_mismatches.sort(
            key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}.get(x.get('severity', 'MEDIUM').upper(), 1)
        )
        
        return {
            "bill_id": bill_facts.get('bill_id'),
            "category": bill_category,
            "matched_category": matched_category if category_valid else None,
            "category_similarity": similarity,
            "mismatches": deduplicated_mismatches,
            "total_rules_checked": len(relevant_rules),
            "violation_count": len(deduplicated_mismatches),
            "is_compliant": len(deduplicated_mismatches) == 0
        }
    
    def _is_deterministically_checkable(self, rule: Dict[str, Any]) -> bool:
        """Check if rule has any attributes that can be validated deterministically"""
        attrs = rule.get('attributes', {})
        return bool(attrs and any(v is not None for v in attrs.values()))
    
    def _run_deterministic_checks(
        self, 
        bill_facts: Dict[str, Any], 
        rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        GENERIC deterministic validation for ANY rule attribute.
        Automatically infers comparison operator from parameter name.
        """
        violations = []
        bill_meta = bill_facts.get('bill_meta', {})
        
        # Extract bill fields
        bill_amount = bill_meta.get('amount')
        bill_currency = bill_meta.get('currency', 'INR').upper()
        bill_date_str = bill_meta.get('date')
        bill_mode = bill_meta.get('mode', '').lower() if bill_meta.get('mode') else ''
        bill_quantity = bill_meta.get('quantity')
        
        # Calculate bill age if date available
        bill_age_days = None
        if bill_date_str:
            try:
                bill_date = datetime.fromisoformat(bill_date_str.replace('Z', '+00:00'))
                if bill_date.tzinfo is None:
                    bill_date = bill_date.replace(tzinfo=timezone.utc)
                current_date = datetime.now(timezone.utc)
                bill_age_days = (current_date - bill_date).days
            except Exception as e:
                logger.warning(f"Could not parse bill date '{bill_date_str}': {e}")
        
        logger.info(f"Bill fields: currency={bill_currency}, amount={bill_amount}, age={bill_age_days} days")
        
        for rule in rules:
            attrs = rule.get('attributes', {})
            rule_id = rule.get('rule_id', 'unknown')
            rule_text = rule.get('raw_text', '')
            rule_category = rule.get('category', '')
            
            if not attrs:
                continue  # No attributes to check
            
            # Iterate through each attribute and check generically
            for attr_name, attr_value in attrs.items():
                if attr_value is None:
                    continue
                
                violation = self._generic_compare(
                    attr_name=attr_name,
                    attr_value=attr_value,
                    bill_meta=bill_meta,
                    bill_amount=bill_amount,
                    bill_currency=bill_currency,
                    bill_age_days=bill_age_days,
                    bill_date_str=bill_date_str,
                    bill_mode=bill_mode,
                    bill_quantity=bill_quantity,
                    rule_id=rule_id,
                    rule_text=rule_text,
                    rule_category=rule_category,
                    rule_severity=rule.get('severity', 'MEDIUM')
                )
                
                if violation:
                    violations.append(violation)
                    break  # Stop checking other attributes for this rule
        
        logger.info(f"Deterministic checks found {len(violations)} violations")
        return violations
    
    def _generic_compare(
        self,
        attr_name: str,
        attr_value: Any,
        bill_meta: Dict[str, Any],
        bill_amount: Any,
        bill_currency: str,
        bill_age_days: Optional[int],
        bill_date_str: Optional[str],
        bill_mode: str,
        bill_quantity: Any,
        rule_id: str,
        rule_text: str,
        rule_category: str,
        rule_severity: str
    ) -> Optional[Dict[str, Any]]:
        """
        GENERIC comparison logic that infers operator from attribute name.
        Works for ANY rule attribute without hardcoding.
        
        Naming conventions (inferred automatically):
        - max_* → check bill value <= attr value
        - min_* → check bill value >= attr value
        - allowed_* → check bill value in list
        - disallowed_* → check bill value not in list
        - *_limit_days → check days old <= attr value
        - currency → check equality
        - (others) → check equality
        """
        
        # Determine comparison type by attribute name
        attr_name_lower = attr_name.lower()
        
        # MAX checks (max_amount, max_quantity, etc.)
        if attr_name_lower.startswith('max_'):
            field_name = attr_name_lower.replace('max_', '')
            
            if 'amount' in field_name:
                bill_value = bill_amount
            elif 'quantity' in field_name:
                bill_value = bill_quantity
            else:
                bill_value = bill_meta.get(attr_name, bill_meta.get(field_name))
            
            if bill_value is not None:
                try:
                    if float(bill_value) > float(attr_value):
                        return {
                            "rule_id": rule_id,
                            "compliant": False,
                            "classification": f"{field_name.title()} Exceeded",
                            "severity": rule_severity,
                            "explanation": f"Bill {attr_name} ({bill_value}) exceeds maximum ({attr_value})",
                            "confidence": 1.0,
                            "violation_details": {
                                "expected": f"<= {attr_value}",
                                "actual": str(bill_value),
                                "deviation": f"{float(bill_value) - float(attr_value)} over limit"
                            },
                            "company_rule_text": rule_text,
                            "rule_category": rule_category,
                            "bill_snippet": f"{attr_name}: {bill_value}",
                            "model_used": "deterministic_check"
                        }
                except (ValueError, TypeError):
                    pass
        
        # MIN checks (min_amount, min_quantity, etc.)
        elif attr_name_lower.startswith('min_'):
            field_name = attr_name_lower.replace('min_', '')
            
            if 'amount' in field_name:
                bill_value = bill_amount
            elif 'quantity' in field_name:
                bill_value = bill_quantity
            else:
                bill_value = bill_meta.get(attr_name, bill_meta.get(field_name))
            
            if bill_value is not None:
                try:
                    if float(bill_value) < float(attr_value):
                        return {
                            "rule_id": rule_id,
                            "compliant": False,
                            "classification": f"{field_name.title()} Below Minimum",
                            "severity": rule_severity,
                            "explanation": f"Bill {attr_name} ({bill_value}) is below minimum ({attr_value})",
                            "confidence": 1.0,
                            "violation_details": {
                                "expected": f">= {attr_value}",
                                "actual": str(bill_value),
                                "deviation": f"{float(attr_value) - float(bill_value)} below minimum"
                            },
                            "company_rule_text": rule_text,
                            "rule_category": rule_category,
                            "bill_snippet": f"{attr_name}: {bill_value}",
                            "model_used": "deterministic_check"
                        }
                except (ValueError, TypeError):
                    pass
        
        # TIME LIMIT checks (*_limit_days, *_limit_days, etc.)
        elif 'limit_days' in attr_name_lower or attr_name_lower.endswith('_limit_days'):
            if bill_age_days is not None:
                try:
                    if bill_age_days > float(attr_value):
                        return {
                            "rule_id": rule_id,
                            "compliant": False,
                            "classification": "Submission Deadline Exceeded",
                            "severity": rule_severity,
                            "explanation": f"Bill submitted {bill_age_days} days after expense date, exceeds {attr_value} day limit",
                            "confidence": 1.0,
                            "violation_details": {
                                "expected": f"<= {attr_value} days",
                                "actual": f"{bill_age_days} days",
                                "deviation": f"{bill_age_days - int(attr_value)} days late"
                            },
                            "company_rule_text": rule_text,
                            "rule_category": rule_category,
                            "bill_snippet": f"Bill Date: {bill_date_str}, Age: {bill_age_days} days",
                            "model_used": "deterministic_check"
                        }
                except (ValueError, TypeError):
                    pass
        
        # ALLOWED values checks (allowed_modes, allowed_payment_methods, etc.)
        elif attr_name_lower.startswith('allowed_'):
            field_name = attr_name_lower.replace('allowed_', '')
            
            # Try to match bill field
            bill_value = bill_meta.get(field_name, bill_meta.get(attr_name))
            
            if bill_value is not None:
                allowed_list = attr_value if isinstance(attr_value, list) else [attr_value]
                if bill_value not in allowed_list:
                    return {
                        "rule_id": rule_id,
                        "compliant": False,
                        "classification": f"Not in allowed {field_name}",
                        "severity": rule_severity,
                        "explanation": f"Bill {field_name} ({bill_value}) not in allowed list: {allowed_list}",
                        "confidence": 1.0,
                        "violation_details": {
                            "expected": str(allowed_list),
                            "actual": str(bill_value),
                            "deviation": "Value not in allowed list"
                        },
                        "company_rule_text": rule_text,
                        "rule_category": rule_category,
                        "bill_snippet": f"{field_name}: {bill_value}",
                        "model_used": "deterministic_check"
                    }
        
        # DISALLOWED values checks (disallowed_modes, disallowed_payment_methods, etc.)
        elif attr_name_lower.startswith('disallowed_'):
            field_name = attr_name_lower.replace('disallowed_', '')
            
            # Try to match bill field
            bill_value = bill_meta.get(field_name, bill_meta.get(attr_name))
            
            if bill_value is not None:
                disallowed_list = attr_value if isinstance(attr_value, list) else [attr_value]
                if bill_value in disallowed_list:
                    return {
                        "rule_id": rule_id,
                        "compliant": False,
                        "classification": f"Disallowed {field_name}",
                        "severity": rule_severity,
                        "explanation": f"Bill {field_name} ({bill_value}) is in disallowed list: {disallowed_list}",
                        "confidence": 1.0,
                        "violation_details": {
                            "expected": f"not in {disallowed_list}",
                            "actual": str(bill_value),
                            "deviation": "Value in disallowed list"
                        },
                        "company_rule_text": rule_text,
                        "rule_category": rule_category,
                        "bill_snippet": f"{field_name}: {bill_value}",
                        "model_used": "deterministic_check"
                    }
        
        # EQUALITY/DEFAULT checks (currency, category, etc.)
        else:
            # Try to find matching bill field
            bill_value = bill_meta.get(attr_name)
            
            if bill_value is not None:
                # Normalize for comparison
                attr_normalized = str(attr_value).lower()
                bill_normalized = str(bill_value).lower()
                
                if attr_normalized != bill_normalized:
                    return {
                        "rule_id": rule_id,
                        "compliant": False,
                        "classification": f"{attr_name.replace('_', ' ').title()} Mismatch",
                        "severity": rule_severity,
                        "explanation": f"Bill {attr_name} ({bill_value}) does not match required {attr_name} ({attr_value})",
                        "confidence": 1.0,
                        "violation_details": {
                            "expected": str(attr_value),
                            "actual": str(bill_value),
                            "deviation": f"{attr_name} mismatch"
                        },
                        "company_rule_text": rule_text,
                        "rule_category": rule_category,
                        "bill_snippet": f"{attr_name}: {bill_value}",
                        "model_used": "deterministic_check"
                    }
        
        return None
    
    def _validate_category(
        self, 
        bill_category: str, 
        stored_categories: List[str]
    ) -> Tuple[bool, Optional[str], int]:
        """Validates category with fuzzy matching"""
        if not bill_category or not stored_categories:
            return False, None, 0
        
        bill_category_lower = bill_category.lower()
        
        # Direct match
        for cat in stored_categories:
            if bill_category_lower == cat.lower():
                return True, cat, 100
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for cat in stored_categories:
            score = fuzz.ratio(bill_category_lower, cat.lower())
            if score > best_score:
                best_score = score
                best_match = cat
        
        is_valid = best_score >= self.CATEGORY_MATCH_THRESHOLD
        return is_valid, best_match, best_score
    
    def _construct_bill_text(self, bill_facts: Dict[str, Any]) -> str:
        """Construct bill text from metadata"""
        bill_meta = bill_facts.get('bill_meta', {})
        
        parts = [
            f"Category: {bill_meta.get('category', 'N/A')}",
            f"Amount: {bill_meta.get('amount', 'N/A')} {bill_meta.get('currency', 'INR')}",
        ]
        
        if bill_meta.get('mode'):
            parts.append(f"Mode: {bill_meta.get('mode')}")
        if bill_meta.get('vendor'):
            parts.append(f"Vendor: {bill_meta.get('vendor')}")
        if bill_meta.get('date'):
            parts.append(f"Date: {bill_meta.get('date')}")
        if bill_meta.get('description'):
            parts.append(f"Description: {bill_meta.get('description')}")
        
        return ". ".join(parts)
    
    def _deduplicate_violations(self, mismatches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced deduplication with special handling for currency violations"""
        if len(mismatches) <= 1:
            return mismatches
        
        unique_map = {}
        
        # Special handling: Keep only ONE currency violation
        currency_violations = [m for m in mismatches if m.get('classification') == 'Currency Mismatch']
        other_violations = [m for m in mismatches if m.get('classification') != 'Currency Mismatch']
        
        # If multiple currency violations, keep the one with most explicit rule text
        if currency_violations:
            best_currency_violation = max(
                currency_violations,
                key=lambda v: (
                    'reimbursement' in v.get('company_rule_text', '').lower() or
                    'processed in' in v.get('company_rule_text', '').lower() or
                    'submitted in' in v.get('company_rule_text', '').lower()
                )
            )
            other_violations.append(best_currency_violation)
            logger.info(f"Deduplicated {len(currency_violations)} currency violations to 1")
        
        # Now deduplicate other violations normally
        for m in other_violations:
            classification = m.get('classification', 'Unknown')
            rule_text = m.get('company_rule_text', '').strip()
            violation_details = m.get('violation_details', {})
            
            # Create signature
            sig = (
                f"{classification}||"
                f"{rule_text}||"
                f"{violation_details.get('expected', '')}||"
                f"{violation_details.get('actual', '')}"
            )
            
            confidence = m.get('confidence', 0.5)
            
            if sig not in unique_map or confidence > unique_map[sig].get('confidence', 0):
                unique_map[sig] = m
        
        result = list(unique_map.values())
        
        if len(result) < len(mismatches):
            logger.info(f"Deduplication: {len(mismatches)} → {len(result)} violations")
        
        return result
    
    def calculate_score(self, compliance_result: Dict[str, Any]) -> int:
        """Calculate compliance score"""
        if compliance_result.get('is_compliant', False):
            return 100
        
        mismatches = compliance_result.get('mismatches', [])
        if not mismatches:
            return 100
        
        severity_weights = {
            'HIGH': 30,
            'MEDIUM': 15,
            'LOW': 5,
        }
        
        total_penalty = sum(
            severity_weights.get(m.get('severity', 'MEDIUM').upper(), 15) * m.get('confidence', 0.8)
            for m in mismatches
        )
        
        return max(0, 100 - min(100, int(total_penalty)))
    
    def generate_detailed_report(self, compliance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        mismatches = compliance_result.get('mismatches', [])
        
        high_severity = [m for m in mismatches if m.get('severity') == 'HIGH']
        medium_severity = [m for m in mismatches if m.get('severity') == 'MEDIUM']
        low_severity = [m for m in mismatches if m.get('severity') == 'LOW']
        
        classifications = {}
        for m in mismatches:
            cls = m.get('classification', 'Unknown')
            classifications[cls] = classifications.get(cls, 0) + 1
        
        score = self.calculate_score(compliance_result)
        
        summary = ""
        if not mismatches:
            summary = "✅ Bill is fully compliant with company policy."
        else:
            summary = f"⚠️ Found {len(mismatches)} policy violation(s):\n"
            if high_severity:
                summary += f"  • {len(high_severity)} HIGH severity issue(s)\n"
            if medium_severity:
                summary += f"  • {len(medium_severity)} MEDIUM severity issue(s)\n"
            if low_severity:
                summary += f"  • {len(low_severity)} LOW severity issue(s)\n"
        
        return {
            "bill_id": compliance_result.get('bill_id'),
            "compliance_score": score,
            "is_compliant": compliance_result.get('is_compliant', False),
            "total_violations": len(mismatches),
            "severity_breakdown": {
                "high": len(high_severity),
                "medium": len(medium_severity),
                "low": len(low_severity)
            },
            "classification_breakdown": classifications,
            "summary": summary.strip(),
            "violations": mismatches,
            "category_info": {
                "bill_category": compliance_result.get('category'),
                "matched_category": compliance_result.get('matched_category'),
                "similarity": compliance_result.get('category_similarity')
            }
        }