# services/improved_compliance_checker.py - Comprehensive compliance checking

from typing import List, Dict, Any, Tuple, Optional
import logging
from services.rag_engine import RAGEngine
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

class ComplianceChecker:
    """
    Enhanced compliance checker that:
    - Validates category existence with fuzzy matching
    - Performs comprehensive LLM-based analysis
    - Detects multiple violations per bill
    - No hardcoded deterministic checks
    """
    
    # Category similarity threshold
    CATEGORY_MATCH_THRESHOLD = 80  # 0-100
    
    def __init__(self):
        self.rag_engine = RAGEngine()
        logger.info("Improved Compliance Checker initialized")
    
    def check_compliance(
        self,
        bill_facts: Dict[str, Any],
        company: str,
        stored_categories: List[str],
        policy_name: Optional[str] = None  # ✅ new parameter
    ) -> Dict[str, Any]:
        """
        Comprehensive compliance check:
        1. Validate category exists
        2. Retrieve relevant rules
        3. LLM analysis for ALL rules
        4. Aggregate all violations
        """
        
        mismatches = []
        bill_meta = bill_facts.get('bill_meta', {})
        bill_category = bill_facts.get("category", bill_meta.get("category", "")).strip()
        
        # Step 1: Category validation with fuzzy matching
        category_valid, matched_category, similarity = self._validate_category(
            bill_category, 
            stored_categories
        )
        
        if not category_valid:
            mismatches.append({
                "classification": "Unrecognized Category",
                "severity": "HIGH",
                "explanation": f"Bill category '{bill_category}' does not match any policy categories. Closest match is '{matched_category}' with {similarity}% similarity (threshold: {self.CATEGORY_MATCH_THRESHOLD}%). Valid categories: {', '.join(stored_categories)}",
                "confidence": 0.99,
                "company_rule_text": "Category validation check",
                "bill_snippet": f"Category: {bill_category}",
                "suggested_category": matched_category if matched_category else None
            })
            # Continue checking even if category is invalid
        
        # Update bill with matched category if fuzzy match found
        if matched_category and similarity >= self.CATEGORY_MATCH_THRESHOLD:
            bill_facts['category'] = matched_category
            bill_meta['category'] = matched_category
        
        # Step 2: Retrieve relevant rules using RAG
        bill_text = bill_facts.get('raw_text', '')
        if not bill_text:
            # Construct text from metadata if raw text not available
            bill_text = self._construct_bill_text(bill_facts)
        
        bill_embedding = self.rag_engine.generate_embedding(bill_text)
        
        relevant_rules = self.rag_engine.retrieve_relevant_rules(
            company=company,
            bill_embedding=bill_embedding,
            bill_facts=bill_facts, 
            top_k=10,  # Retrieve more rules for comprehensive checking
            policy_name=policy_name
        )
        
        if not relevant_rules:
            logger.warning(f"No rules found for company: {company}, category: {bill_category}")
            mismatches.append({
                "classification": "No Policy Rules Found",
                "severity": "MEDIUM",
                "explanation": f"No policy rules found for category '{bill_category}'. Unable to verify compliance.",
                "confidence": 0.95,
                "company_rule_text": "N/A",
                "bill_snippet": bill_text[:200]
            })
        
        # Step 3: LLM-based analysis for EACH rule
        logger.info(f"Checking bill against {len(relevant_rules)} rules")
        
        for rule in relevant_rules:
            try:
                # Use LLM to reason about this specific rule
                llm_result = self.rag_engine.reason_with_llm(bill_facts, rule)
                
                # Add violation if non-compliant
                if not llm_result.get('compliant', True):
                    # Enrich with rule metadata
                    llm_result['rule_id'] = rule.get('rule_id')
                    llm_result['rule_category'] = rule.get('category')
                    mismatches.append(llm_result)
                    
            except Exception as e:
                logger.error(f"Error checking rule {rule.get('rule_id')}: {e}")
                continue
        
        # Step 4: Deduplicate similar violations
        deduplicated_mismatches = self._deduplicate_violations(mismatches)
        
        # Step 5: Sort by severity
        deduplicated_mismatches.sort(
            key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}.get(x.get('severity', 'MEDIUM'), 1)
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
    
    def _validate_category(
        self, 
        bill_category: str, 
        stored_categories: List[str]
    ) -> Tuple[bool, str, int]:
        """
        Validate category with fuzzy matching.
        Returns: (is_valid, best_match, similarity_score)
        """
        if not bill_category or not stored_categories:
            return False, None, 0
        
        # Exact match
        if bill_category in stored_categories:
            return True, bill_category, 100
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for cat in stored_categories:
            # Use multiple fuzzy matching algorithms
            ratio_score = fuzz.ratio(bill_category.lower(), cat.lower())
            partial_score = fuzz.partial_ratio(bill_category.lower(), cat.lower())
            token_sort_score = fuzz.token_sort_ratio(bill_category.lower(), cat.lower())
            
            # Take the best score
            score = max(ratio_score, partial_score, token_sort_score)
            
            if score > best_score:
                best_score = score
                best_match = cat
        
        is_valid = best_score >= self.CATEGORY_MATCH_THRESHOLD
        
        return is_valid, best_match, best_score
    
    def _construct_bill_text(self, bill_facts: Dict[str, Any]) -> str:
        """Construct readable bill text from metadata."""
        bill_meta = bill_facts.get('bill_meta', {})
        
        parts = []
        parts.append(f"Category: {bill_meta.get('category', 'N/A')}")
        parts.append(f"Amount: {bill_meta.get('amount', 'N/A')} {bill_meta.get('currency', 'INR')}")
        
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
        """
        Remove duplicate or very similar violations.
        Keeps the violation with highest confidence.
        """
        if len(mismatches) <= 1:
            return mismatches
        
        unique_violations = []
        seen_classifications = {}
        
        for mismatch in mismatches:
            classification = mismatch.get('classification', 'Unknown')
            confidence = mismatch.get('confidence', 0.5)
            
            # Create a key based on classification
            key = f"{classification}"
            
            if key not in seen_classifications:
                seen_classifications[key] = mismatch
                unique_violations.append(mismatch)
            else:
                # If we've seen this classification, keep the one with higher confidence
                existing = seen_classifications[key]
                if confidence > existing.get('confidence', 0):
                    # Replace with higher confidence version
                    unique_violations.remove(existing)
                    unique_violations.append(mismatch)
                    seen_classifications[key] = mismatch
        
        return unique_violations
    
    def batch_check_compliance(
        self,
        bills: List[Dict[str, Any]],
        company: str,
        stored_categories: List[str]
    ) -> List[Dict[str, Any]]:
        """Check compliance for multiple bills."""
        results = []
        
        for bill in bills:
            try:
                result = self.check_compliance(bill, company, stored_categories)
                results.append(result)
            except Exception as e:
                logger.error(f"Error checking bill {bill.get('bill_id')}: {e}")
                results.append({
                    "bill_id": bill.get('bill_id'),
                    "error": str(e),
                    "is_compliant": False
                })
        
        return results
    
    def calculate_score(self, compliance_result: Dict[str, Any]) -> int:
        """
        Calculate compliance score (0-100).
        Considers severity and confidence.
        """
        if compliance_result.get('is_compliant', False):
            return 100
        
        mismatches = compliance_result.get('mismatches', [])
        if not mismatches:
            return 100
        
        # Severity penalties
        severity_weights = {
            'HIGH': 25,
            'MEDIUM': 12,
            'LOW': 5
        }
        
        total_penalty = 0
        for mismatch in mismatches:
            severity = mismatch.get('severity', 'MEDIUM')
            confidence = mismatch.get('confidence', 0.8)
            
            base_penalty = severity_weights.get(severity, 12)
            weighted_penalty = base_penalty * confidence
            total_penalty += weighted_penalty
        
        # Cap penalty at 100
        total_penalty = min(100, total_penalty)
        score = max(0, 100 - int(total_penalty))
        
        return score
    
    def generate_detailed_report(self, compliance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        mismatches = compliance_result.get('mismatches', [])
        
        # Categorize violations
        high_severity = [m for m in mismatches if m.get('severity') == 'HIGH']
        medium_severity = [m for m in mismatches if m.get('severity') == 'MEDIUM']
        low_severity = [m for m in mismatches if m.get('severity') == 'LOW']
        
        # Classification breakdown
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