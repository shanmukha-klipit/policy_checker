# database/mongodb_client.py - MongoDB operations with vector search

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    MongoDB client for storing policies, bills, and compliance results.
    Supports vector search for RAG functionality with local embeddings.
    """
    
    def __init__(self):
        # MongoDB connection
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(mongo_uri)
        self.db = self.client['compliance_checker']
        
        # Collections
        self.policies = self.db['policies']
        self.rules = self.db['policy_rules']
        self.bills = self.db['bills']
        self.compliance_checks = self.db['compliance_checks']
        
        # Create indexes
        self._create_indexes()
        
        logger.info("MongoDB client initialized")
    
    def _create_indexes(self):
        """Create necessary indexes for efficient queries."""
        try:
            # Policy rules indexes
            self.rules.create_index([("company", ASCENDING)])
            self.rules.create_index([("category", ASCENDING)])
            self.rules.create_index([("company", ASCENDING), ("category", ASCENDING)])
            
            # Bills indexes
            self.bills.create_index([("bill_id", ASCENDING)], unique=True)
            self.bills.create_index([("company", ASCENDING)])
            
            # Compliance checks indexes
            self.compliance_checks.create_index([("company", ASCENDING)])
            self.compliance_checks.create_index([("timestamp", DESCENDING)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def store_policy_rules(self, company: str, rules: List[Dict[str, Any]]) -> bool:
        """
        Store extracted policy rules with embeddings.
        """
        try:
            # Add company identifier to each rule
            for rule in rules:
                rule['company'] = company
                rule['created_at'] = datetime.utcnow().isoformat()
            
            # Delete existing rules for this company (policy update)
            self.rules.delete_many({"company": company})
            
            # Insert new rules
            if rules:
                self.rules.insert_many(rules)
                logger.info(f"Stored {len(rules)} rules for {company}")

            allowed_categories = sorted(list({rule.get('category', 'Other') for rule in rules}))
            
            # Also store policy metadata
            policy_doc = {
                "company": company,
                "rules_count": len(rules),
                "allowed_categories": allowed_categories,
                "uploaded_at": datetime.utcnow().isoformat(),
                "version": datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            }
            
            self.policies.replace_one(
                {"company": company},
                policy_doc,
                upsert=True
            )
            
            return True
        except Exception as e:
            logger.error(f"Error storing policy rules: {e}")
            return False

    def get_allowed_categories(self, company: str) -> List[str]:
        """
        Retrieve allowed categories defined in a company's uploaded policy.
        """
        try:
            policy = self.policies.find_one({"company": company}, {"allowed_categories": 1, "_id": 0})
            if policy and "allowed_categories" in policy:
                return policy["allowed_categories"]
            else:
                logger.warning(f"No allowed categories found for {company}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving allowed categories for {company}: {e}")
            return []

    def store_bill(self, bill_data: Dict[str, Any]) -> bool:
        """Store parsed bill data."""
        try:
            bill_data['stored_at'] = datetime.utcnow().isoformat()
            
            self.bills.replace_one(
                {"bill_id": bill_data['bill_id']},
                bill_data,
                upsert=True
            )
            
            logger.info(f"Stored bill: {bill_data['bill_id']}")
            return True
        except Exception as e:
            logger.error(f"Error storing bill: {e}")
            return False
    
    def store_compliance_check(self, check_data: Dict[str, Any]) -> bool:
        """Store compliance check result."""
        try:
            check_data['_id'] = f"{check_data['company']}_{datetime.utcnow().timestamp()}"
            check_data['timestamp'] = datetime.utcnow().isoformat()
            
            self.compliance_checks.insert_one(check_data)
            
            logger.info(f"Stored compliance check for {check_data['company']}")
            return True
        except DuplicateKeyError:
            logger.warning("Duplicate compliance check, updating...")
            self.compliance_checks.replace_one(
                {"_id": check_data['_id']},
                check_data
            )
            return True
        except Exception as e:
            logger.error(f"Error storing compliance check: {e}")
            return False
    
    def get_compliance_history(self, company: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent compliance check history for a company."""
        try:
            history = list(
                self.compliance_checks.find(
                    {"company": company},
                    {"_id": 0}
                ).sort("timestamp", DESCENDING).limit(limit)
            )
            return history
        except Exception as e:
            logger.error(f"Error retrieving compliance history: {e}")
            return []
    
    def delete_policy(self, company: str) -> bool:
        """Delete all policy data for a company."""
        try:
            self.rules.delete_many({"company": company})
            self.policies.delete_one({"company": company})
            
            logger.info(f"Deleted policy data for {company}")
            return True
        except Exception as e:
            logger.error(f"Error deleting policy: {e}")
            return False
    
    def get_statistics(self, company: str) -> Dict[str, Any]:
        """Get compliance statistics for a company."""
        try:
            # Count rules
            rules_count = self.rules.count_documents({"company": company})
            
            # Count checks
            checks_count = self.compliance_checks.count_documents({"company": company})
            
            # Average compliance score
            pipeline = [
                {"$match": {"company": company}},
                {"$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$score"},
                    "total_violations": {"$sum": {"$size": "$result.mismatches"}}
                }}
            ]
            
            agg_result = list(self.compliance_checks.aggregate(pipeline))
            
            stats = {
                "company": company,
                "total_rules": rules_count,
                "total_checks": checks_count,
                "avg_score": round(agg_result[0]['avg_score'], 2) if agg_result else 0,
                "total_violations": agg_result[0]['total_violations'] if agg_result else 0
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def search_similar_rules(
        self,
        company: str,
        embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar rules using vector similarity.
        Works with sentence-transformers embeddings (384 dimensions).
        Handles dimension mismatches gracefully.
        """
        try:
            # Get all rules for the company
            all_rules = self.get_policy_rules(company)
            
            if not all_rules:
                logger.warning(f"No rules found for company: {company}")
                return []
            
            # Calculate cosine similarity
            import numpy as np
            
            scored_rules = []
            query_vec = np.array(embedding)
            
            for rule in all_rules:
                # Check if rule has valid embedding
                if 'embedding' not in rule or not rule['embedding']:
                    logger.debug(f"Rule {rule.get('rule_id', 'unknown')} has no embedding, skipping")
                    continue
                
                if len(rule['embedding']) == 0:
                    logger.debug(f"Rule {rule.get('rule_id', 'unknown')} has empty embedding, skipping")
                    continue
                
                rule_vec = np.array(rule['embedding'])
                
                # Handle dimension mismatch gracefully
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
            
            # If no rules with valid embeddings, return first top_k rules without scoring
            if not scored_rules:
                logger.warning("No rules with valid embeddings found, returning first rules without similarity scoring")
                return all_rules[:top_k]
            
            # Sort by similarity (highest first)
            scored_rules.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            logger.info(f"Found {len(scored_rules)} rules with similarity scores")
            return scored_rules[:top_k]
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            # Fallback: return rules without similarity scoring
            try:
                all_rules = self.get_policy_rules(company)
                return all_rules[:top_k]
            except:
                return []
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")


    def store_company_categories(self, company: str, categories: List[str]):
        """Store categories for a company"""
        try:
            self.db.companies.update_one(
                {"company": company},
                {"$set": {
                    "categories": categories,
                    "last_updated": datetime.utcnow().isoformat()
                }},
                upsert=True
            )
            logger.info(f"Stored categories for company: {company}")
            return True
        except Exception as e:
            logger.error(f"Error storing company categories for {company}: {e}")
            return False
