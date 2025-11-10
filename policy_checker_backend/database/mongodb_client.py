# database/mongodb_client.py - MongoDB operations with restructured schema

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    MongoDB client for storing policies and compliance checks.
    Uses two collections:
    1. policy_rules - Stores complete policy information with rules, embeddings, categories
    2. compliance_checks - Stores compliance check results with violations
    """
    
    def __init__(self, db_name: str = None):
        # MongoDB connection
        mongo_uri = os.getenv("MONGODB_URI")
        self.client = MongoClient(mongo_uri)
        
        # Use provided db_name or default to 'klipit'
        db_name = db_name or os.getenv("MONGODB_DB_NAME")
        self.db = self.client[db_name]
        
        # Collections - only two now
        self.policy_rules = self.db['policy_rules']
        self.compliance_checks = self.db['compliance_checks']
        
        # Create indexes
        self._create_indexes()
        
        logger.info(f"MongoDB client initialized with database: {db_name}")
    
    def _create_indexes(self):
        """Create necessary indexes for efficient queries."""
        try:
            # Policy rules indexes
            self.policy_rules.create_index([("company", ASCENDING)])
            self.policy_rules.create_index([("policy_name", ASCENDING)])
            self.policy_rules.create_index([("company", ASCENDING), ("policy_name", ASCENDING)], unique=True)
            self.policy_rules.create_index([("effective_from", DESCENDING)])
            self.policy_rules.create_index([("time_uploaded", DESCENDING)])
            
            # Compliance checks indexes
            self.compliance_checks.create_index([("company", ASCENDING)])
            self.compliance_checks.create_index([("time_uploaded", DESCENDING)])
            self.compliance_checks.create_index([("company", ASCENDING), ("time_uploaded", DESCENDING)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def store_policy(self, policy_data: Dict[str, Any]) -> bool:

        """
        Store complete policy information in a single document.
        
        Expected policy_data structure:
        {
            "company": str,
            "file_path": str,
            "policy_name": str,
            "description": str (optional),  # NEW: Added description
            "rules_extracted": [
                {
                    "rule_id": str,
                    "rule_text": str,
                    "category": str,
                    "embedding": List[float],
                    "conditions": List[str],
                    "amount_limit": float (optional),
                    ...
                }
            ],
            "effective_from": str (ISO format),
            "effective_to": str (ISO format, optional),
            "categories": List[str],
            "embeddings_model": str (optional),
            "total_rules": int,
            "version": str (optional)
        }
        """
        try:
            # Add metadata
            current_time = datetime.utcnow()
            policy_data['time_uploaded'] = current_time.isoformat()
            policy_data['last_updated'] = current_time.isoformat()
            
            # Add computed fields if not present
            if 'total_rules' not in policy_data:
                policy_data['total_rules'] = len(policy_data.get('rules_extracted', []))
            
            if 'status' not in policy_data:
                # Check if policy is currently active
                effective_from = datetime.fromisoformat(policy_data.get('effective_from', current_time.isoformat()))
                effective_to = policy_data.get('effective_to')
                
                if effective_to:
                    effective_to_dt = datetime.fromisoformat(effective_to)
                    if current_time < effective_from:
                        policy_data['status'] = 'scheduled'
                    elif current_time > effective_to_dt:
                        policy_data['status'] = 'expired'
                    else:
                        policy_data['status'] = 'active'
                else:
                    policy_data['status'] = 'active' if current_time >= effective_from else 'scheduled'
            
            # Extract unique categories from rules if not provided
            if 'categories' not in policy_data or not policy_data['categories']:
                categories = set()
                for rule in policy_data.get('rules_extracted', []):
                    if 'category' in rule:
                        categories.add(rule['category'])
                policy_data['categories'] = sorted(list(categories))
            
            # Store or update policy
            result = self.policy_rules.replace_one(
                {
                    "company": policy_data['company'],
                    "policy_name": policy_data['policy_name']
                },
                policy_data,
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"Created new policy: {policy_data['company']} - {policy_data['policy_name']}")
            else:
                logger.info(f"Updated existing policy: {policy_data['company']} - {policy_data['policy_name']}")
            
            return True
        except Exception as e:
            logger.error(f"Error storing policy: {e}")
            return False
    
    def get_policy(self, company: str, policy_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve policy for a company. If policy_name is not provided, returns the most recent active policy.
        """
        try:
            if policy_name:
                policy = self.policy_rules.find_one({
                    "company": company,
                    "policy_name": policy_name
                })
            else:
                # Get most recent active policy
                policy = self.policy_rules.find_one(
                    {
                        "company": company,
                        "status": "active"
                    },
                    sort=[("time_uploaded", DESCENDING)]
                )
            
            if policy:
                policy.pop('_id', None)  # Remove MongoDB _id
                return policy
            return None
        except Exception as e:
            logger.error(f"Error retrieving policy: {e}")
            return None
    
    def get_policy_rules(self, company: str, policy_name: str = None) -> List[Dict[str, Any]]:
        """
        Get only the rules from a policy.
        """
        try:
            policy = self.get_policy(company, policy_name)
            if policy:
                return policy.get('rules_extracted', [])
            return []
        except Exception as e:
            logger.error(f"Error retrieving policy rules: {e}")
            return []
    
    def get_allowed_categories(self, company: str, policy_name: str = None) -> List[str]:
        """
        Retrieve allowed categories from a company's policy.
        """
        try:
            policy = self.get_policy(company, policy_name)
            if policy:
                return policy.get('categories', [])
            return []
        except Exception as e:
            logger.error(f"Error retrieving categories: {e}")
            return []
    
    def store_compliance_check(self, check_data: Dict[str, Any]) -> bool:
        """
        Store compliance check result.
        
        Expected check_data structure:
        {
            "company": str,
            "file_path": str,
            "violations": [
                {
                    "bill_item_id": str,
                    "violation_type": str,
                    "description": str,
                    "severity": str,
                    "rule_violated": str,
                    ...
                }
            ],
            "classification": {
                "compliant_items": int,
                "non_compliant_items": int,
                "total_items": int,
                "compliance_score": float,
                "categories_checked": List[str]
            },
            "bill_id": str (optional),
            "policy_name": str (optional),
            "checked_by": str (optional),
            "metadata": Dict (optional)
        }
        """
        try:
            current_time = datetime.utcnow()
            
            # Add timestamps
            check_data['time_uploaded'] = current_time.isoformat()
            check_data['check_id'] = f"{check_data['company']}_{int(current_time.timestamp() * 1000)}"
            
            # Add computed fields
            if 'classification' in check_data:
                classification = check_data['classification']
                if 'compliance_score' not in classification and 'total_items' in classification:
                    if classification['total_items'] > 0:
                        classification['compliance_score'] = round(
                            (classification.get('compliant_items', 0) / classification['total_items']) * 100, 
                            2
                        )
            
            # Add violation summary
            if 'violations' in check_data:
                check_data['total_violations'] = len(check_data['violations'])
                
                # Group violations by severity
                severity_count = {}
                for violation in check_data['violations']:
                    severity = violation.get('severity', 'unknown')
                    severity_count[severity] = severity_count.get(severity, 0) + 1
                check_data['violations_by_severity'] = severity_count
            else:
                check_data['total_violations'] = 0
                check_data['violations_by_severity'] = {}
            
            # Insert compliance check
            result = self.compliance_checks.insert_one(check_data)
            
            logger.info(f"Stored compliance check: {check_data['check_id']} for {check_data['company']}")
            return True
        except Exception as e:
            logger.error(f"Error storing compliance check: {e}")
            return False
    
    def get_compliance_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific compliance check by ID.
        """
        try:
            check = self.compliance_checks.find_one({"check_id": check_id})
            if check:
                check.pop('_id', None)
                return check
            return None
        except Exception as e:
            logger.error(f"Error retrieving compliance check: {e}")
            return None
    
    def get_compliance_history(self, company: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent compliance check history for a company.
        """
        try:
            history = list(
                self.compliance_checks.find(
                    {"company": company}
                ).sort("time_uploaded", DESCENDING).limit(limit)
            )
            
            # Remove MongoDB _id from results
            for check in history:
                check.pop('_id', None)
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving compliance history: {e}")
            return []
    
    def get_statistics(self, company: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a company.
        """
        try:
            # Get policy stats
            policy = self.get_policy(company)
            policy_stats = {
                "has_active_policy": policy is not None,
                "total_rules": policy.get('total_rules', 0) if policy else 0,
                "categories": policy.get('categories', []) if policy else [],
                "policy_name": policy.get('policy_name', 'N/A') if policy else 'N/A'
            }
            
            # Get compliance check stats
            total_checks = self.compliance_checks.count_documents({"company": company})
            
            # Average compliance score and total violations
            pipeline = [
                {"$match": {"company": company}},
                {"$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$classification.compliance_score"},
                    "total_violations": {"$sum": "$total_violations"},
                    "avg_violations": {"$avg": "$total_violations"}
                }}
            ]
            
            agg_result = list(self.compliance_checks.aggregate(pipeline))
            
            compliance_stats = {
                "total_checks": total_checks,
                "avg_compliance_score": round(agg_result[0]['avg_score'], 2) if agg_result and agg_result[0]['avg_score'] else 0,
                "total_violations": agg_result[0]['total_violations'] if agg_result else 0,
                "avg_violations_per_check": round(agg_result[0]['avg_violations'], 2) if agg_result and agg_result[0]['avg_violations'] else 0
            }
            
            # Get latest check info
            latest_check = self.compliance_checks.find_one(
                {"company": company},
                sort=[("time_uploaded", DESCENDING)]
            )
            
            latest_check_info = {
                "last_check_time": latest_check.get('time_uploaded', 'N/A') if latest_check else 'N/A',
                "last_check_score": latest_check.get('classification', {}).get('compliance_score', 0) if latest_check else 0
            }
            
            return {
                "company": company,
                "policy": policy_stats,
                "compliance": compliance_stats,
                "latest_check": latest_check_info
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def search_similar_rules(
        self,
        company: str,
        embedding: List[float],
        top_k: int = 5,
        policy_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar rules using vector similarity within a company's policy.
        """
        try:
            import numpy as np
            
            # Get policy rules
            rules = self.get_policy_rules(company, policy_name)
            
            if not rules:
                logger.warning(f"No rules found for company: {company}")
                return []
            
            # Calculate cosine similarity
            scored_rules = []
            query_vec = np.array(embedding)
            
            for rule in rules:
                # Check if rule has valid embedding
                if 'embedding' not in rule or not rule['embedding']:
                    continue
                
                rule_vec = np.array(rule['embedding'])
                
                # Handle dimension mismatch
                if len(query_vec) != len(rule_vec):
                    logger.warning(
                        f"Embedding dimension mismatch for rule {rule.get('rule_id', 'unknown')}: "
                        f"query={len(query_vec)} vs rule={len(rule_vec)}, skipping"
                    )
                    continue
                
                # Calculate cosine similarity
                try:
                    similarity = np.dot(query_vec, rule_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(rule_vec)
                    )
                    rule['similarity'] = float(similarity)
                    scored_rules.append(rule)
                except Exception as sim_error:
                    logger.warning(f"Error calculating similarity: {sim_error}")
                    continue
            
            # If no rules with valid embeddings, return first top_k rules
            if not scored_rules:
                logger.warning("No rules with valid embeddings found")
                return rules[:top_k]
            
            # Sort by similarity (highest first)
            scored_rules.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            logger.info(f"Found {len(scored_rules)} rules with similarity scores")
            return scored_rules[:top_k]
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def delete_policy(self, company: str, policy_name: str = None) -> bool:
        """
        Delete policy data. If policy_name is provided, deletes that specific policy.
        Otherwise, deletes all policies for the company.
        """
        try:
            if policy_name:
                result = self.policy_rules.delete_one({
                    "company": company,
                    "policy_name": policy_name
                })
                logger.info(f"Deleted policy {policy_name} for {company}")
            else:
                result = self.policy_rules.delete_many({"company": company})
                logger.info(f"Deleted all policies for {company}")
            
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting policy: {e}")
            return False
    
    def delete_compliance_checks(self, company: str, older_than_days: int = None) -> bool:
        """
        Delete compliance checks for a company.
        If older_than_days is provided, only deletes checks older than that.
        """
        try:
            query = {"company": company}
            
            if older_than_days:
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
                query["time_uploaded"] = {"$lt": cutoff_date.isoformat()}
            
            result = self.compliance_checks.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} compliance checks for {company}")
            return True
        except Exception as e:
            logger.error(f"Error deleting compliance checks: {e}")
            return False
    
    def list_companies(self) -> List[str]:
        """
        Get list of all companies with policies.
        """
        try:
            companies = self.policy_rules.distinct("company")
            return sorted(companies)
        except Exception as e:
            logger.error(f"Error listing companies: {e}")
            return []
    
    def list_policies(self, company: str) -> List[Dict[str, Any]]:
        """
        List all policies for a company with summary information.
        """
        try:
            policies = list(
                self.policy_rules.find(
                    {"company": company},
                    {
                        "policy_name": 1,
                        "time_uploaded": 1,
                        "effective_from": 1,
                        "effective_to": 1,
                        "status": 1,
                        "total_rules": 1,
                        "categories": 1,
                        "_id": 0
                    }
                ).sort("time_uploaded", DESCENDING)
            )
            return policies
        except Exception as e:
            logger.error(f"Error listing policies: {e}")
            return []

    def get_policies_by_company(self, company: str):
        """
        Fetch all policy documents for a company.
        """
        try:
            # Use policy_rules collection instead of policies
            policies = list(self.policy_rules.find(
                {"company": company},
                {
                    "_id": 1,
                    "policy_name": 1,
                    "description": 1,
                    "status": 1,
                    "effective_from": 1,
                    "effective_to": 1,
                    "categories": 1,
                    "total_rules": 1,
                    "last_updated": 1,  # Changed from updated_at to last_updated
                    "time_uploaded": 1   # Also include time_uploaded as fallback
                }
            ))
            
            # Format the response
            formatted_policies = []
            for policy in policies:
                formatted_policy = {
                    "policy_name": policy.get("policy_name"),
                    "description": policy.get("description", ""),
                    "status": policy.get("status", "active"),
                    "effective_from": policy.get("effective_from"),
                    "effective_to": policy.get("effective_to"),
                    "total_rules": policy.get("total_rules", 0),
                    "categories": policy.get("categories", []),
                    "last_updated": policy.get("last_updated") or policy.get("time_uploaded"),
                }
                formatted_policies.append(formatted_policy)
            
            logger.info(f"Found {len(formatted_policies)} policies for company: {company}")
            return formatted_policies
            
        except Exception as e:
            logger.error(f"Error fetching policies for company={company}: {e}", exc_info=True)
            return []
        
    def get_policy_by_name(self, company: str, policy_name: str):
        try:
            collection = self.db["policy_rules"]
            policy = collection.find_one({"company": company, "policy_name": policy_name})
            return policy
        except Exception as e:
            logging.error(f"Error fetching policy by name: {e}")
            return None
        
    def update_policy(self, company: str, policy_name: str, updated_fields: dict):
        try:
            result = self.db["policy_rules"].update_one(
                {"company": company, "policy_name": policy_name},
                {"$set": updated_fields}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error updating policy: {e}")
            return False

    def get_expense_by_id(self, expense_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an expense from manual_expenses collection by ID.
        
        Args:
            expense_id: The expense document ID (string or ObjectId)
        
        Returns:
            Dict containing expense data or None if not found
        """
        try:
            from bson import ObjectId
            
            # Access manual_expenses collection
            manual_expenses = self.db['manual_expenses']
            
            # Try to convert string ID to ObjectId
            try:
                expense_object_id = ObjectId(expense_id)
                expense = manual_expenses.find_one({"_id": expense_object_id})
            except Exception:
                # If conversion fails, try searching with string ID
                expense = manual_expenses.find_one({"_id": expense_id})
            
            if not expense:
                logger.warning(f"Expense not found with ID: {expense_id}")
                return None
            
            # Convert ObjectId to string for JSON serialization
            if '_id' in expense:
                expense['_id'] = str(expense['_id'])
            
            # Extract relevant fields
            expense_data = {
                "_id": expense.get("_id"),
                "title": expense.get("title"),
                "date": expense.get("date"),
                "currency": expense.get("currency"),
                "originalAmount": expense.get("originalAmount"),
                "totalAmount": expense.get("totalAmount"),
                "convertedCurrency": expense.get("convertedCurrency"),
                "items": expense.get("items", []),
                "receiptId": expense.get("receiptId"),
                "retailer": expense.get("retailer"),
                "time": expense.get("time"),
                "fileUrl": expense.get("fileUrl"),
                "status": expense.get("status"),
                "numberOfItems": expense.get("numberOfItems", 0),
            }
            
            logger.info(f"Retrieved expense: {expense_data.get('title')} with {len(expense_data.get('items', []))} items")
            return expense_data
            
        except Exception as e:
            logger.error(f"Error retrieving expense by ID: {e}", exc_info=True)
            return None

    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")