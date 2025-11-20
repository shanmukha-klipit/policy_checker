# database/mongodb_client.py - Complete corrected implementation

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import os
import logging

logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    MongoDB client with dynamic connection switching for different environments
    """
    
    def __init__(self, db_name: str = None):
        # 🆕 UPDATED: Store environment configurations
        self.environment_configs = {
            'dev': {
                'web_origins': [os.getenv("DB_MAP_DEV_WEB")],
                'mobile_origins': [os.getenv("DB_MAP_DEV_MOBILE")],
                'db_name': os.getenv("DB_NAME_DEV"),
                'mongo_uri': os.getenv("MONGODB_URI_DEV")
            },
            'staging': {
                'web_origins': [os.getenv("DB_MAP_STAGING_WEB")],
                'mobile_origins': [os.getenv("DB_MAP_STAGING_MOBILE")],
                'db_name': os.getenv("DB_NAME_STAGING"),
                'mongo_uri': os.getenv("MONGODB_URI_STAGING")
            },
            'prod': {
                'web_origins': [os.getenv("DB_MAP_PROD_WEB")],
                'mobile_origins': [os.getenv("DB_MAP_PROD_MOBILE")],
                'db_name': os.getenv("DB_NAME_PROD"),
                'mongo_uri': os.getenv("MONGODB_URI_PROD")
            }
        }
        
        # Remove None values
        for env in list(self.environment_configs.keys()):
            config = self.environment_configs[env]
            config = {k: v for k, v in config.items() if v}
            if not config:
                del self.environment_configs[env]
            else:
                self.environment_configs[env] = config
        
        # Default connection (fallback)
        self.default_mongo_uri = os.getenv("MONGODB_URI")
        self.default_db_name = db_name or os.getenv("MONGODB_DB_NAME", "klipit")
        
        # Current connection state
        self.current_env = 'default'
        self.client = None
        self.db = None
        
        # Initialize with default connection
        self._initialize_connection(self.default_mongo_uri, self.default_db_name)
        
        # Collections
        self.policy_rules = self.db['policy_rules']
        self.compliance_checks = self.db['compliance_checks']
        
        # Request metadata
        self.origin = None
        self.referer = None
        self.client_ip = None
        
        # Create indexes
        self._create_indexes()
        
        logger.info(f"✅ MongoDB client initialized with default database: {self.default_db_name}")
        logger.info(f"🌍 Available environments: {list(self.environment_configs.keys())}")
    
    def _initialize_connection(self, mongo_uri: str, db_name: str):
        """Initialize or reinitialize MongoDB connection"""
        try:
            # Close existing connection if any
            if self.client:
                self.client.close()
            
            # Create new connection
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            
            logger.info(f"🔗 MongoDB connected to: {db_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            # Fallback to default connection
            if mongo_uri != self.default_mongo_uri:
                logger.info("🔄 Falling back to default MongoDB connection")
                self.client = MongoClient(self.default_mongo_uri)
                self.db = self.client[self.default_db_name]
            else:
                raise
    
    def _create_indexes(self):
        """Create necessary indexes for efficient queries."""
        try:
            # Policy rules indexes
            self.policy_rules.create_index([("company", ASCENDING)])
            self.policy_rules.create_index([("policy_name", ASCENDING)])
            self.policy_rules.create_index([("company", ASCENDING), ("policy_name", ASCENDING)], unique=True)
            self.policy_rules.create_index([("status", ASCENDING)])
            self.policy_rules.create_index([("time_uploaded", DESCENDING)])
            
            # Compliance checks indexes
            self.compliance_checks.create_index([("company", ASCENDING)])
            self.compliance_checks.create_index([("time_uploaded", DESCENDING)])
            self.compliance_checks.create_index([("company", ASCENDING), ("time_uploaded", DESCENDING)])
            
            logger.info("✅ Database indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
    
    def _detect_environment_from_origin(self, origin: str) -> str:
        """
        Detect environment from origin
        Returns: 'dev', 'staging', 'prod', or 'default'
        """
        if not origin:
            return 'default'
        
        origin_lower = origin.lower()
        
        # Check each environment's origins
        for env, config in self.environment_configs.items():
            # Check web origins
            web_origins = config.get('web_origins', [])
            for web_origin in web_origins:
                if web_origin and web_origin.lower() in origin_lower:
                    return env
            
            # Check mobile origins
            mobile_origins = config.get('mobile_origins', [])
            for mobile_origin in mobile_origins:
                if mobile_origin and mobile_origin.lower() in origin_lower:
                    return env
        
        # Specific pattern matching as fallback
        if "localhost" in origin_lower or "dev" in origin_lower:
            return 'dev'
        elif "staging" in origin_lower:
            return 'staging'
        elif "business.klipit.co" in origin_lower:
            return 'prod'
        
        return 'default'
    
    def switch_db_based_on_origin(self, origin: str):
        """
        🆕 UPDATED: Switch database connection based on origin
        Changes both MongoDB URI and database name
        """
        if not origin:
            logger.warning("⚠️ No origin provided, using default connection")
            return
        
        # Detect environment from origin
        detected_env = self._detect_environment_from_origin(origin)
        
        # If already connected to the correct environment, do nothing
        if detected_env == self.current_env:
            logger.debug(f"✅ Already connected to {detected_env} environment")
            return
        
        # Get environment configuration
        env_config = self.environment_configs.get(detected_env)
        
        if env_config:
            # Switch to environment-specific connection
            mongo_uri = env_config.get('mongo_uri') or self.default_mongo_uri
            db_name = env_config.get('db_name') or self.default_db_name
            
            try:
                self._initialize_connection(mongo_uri, db_name)
                self.current_env = detected_env
                
                # Update collections reference
                self.policy_rules = self.db['policy_rules']
                self.compliance_checks = self.db['compliance_checks']
                
                logger.info(f"✅ Switched to {detected_env.upper()} environment")
                logger.info(f"   Database: {db_name}")
                logger.info(f"   Origin: {origin}")
                
            except Exception as e:
                logger.error(f"❌ Failed to switch to {detected_env} environment: {e}")
                # Fallback to default connection
                self._initialize_connection(self.default_mongo_uri, self.default_db_name)
                self.current_env = 'default'
        else:
            # Use default connection for unknown origins
            if self.current_env != 'default':
                self._initialize_connection(self.default_mongo_uri, self.default_db_name)
                self.current_env = 'default'
                logger.warning(f"⚠️ Unknown origin '{origin}', using default connection")
    
    # 🆕 NEW: Method to get current connection info
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information"""
        return {
            'environment': self.current_env,
            'database': self.db.name if self.db else 'unknown',
            'origin': self.origin
        }

    # ============================================================================
    # EXISTING METHODS - Keep all your existing functionality
    # ============================================================================
    
    def store_policy(self, policy_data: Dict[str, Any]) -> bool:
        """
        Store complete policy information in a single document.
        """
        try:
            # Add metadata
            current_time = datetime.now(timezone.utc)
            policy_data['time_uploaded'] = current_time.isoformat()
            policy_data['last_updated'] = current_time.isoformat()
            
            # Add computed fields if not present
            if 'total_rules' not in policy_data:
                policy_data['total_rules'] = len(policy_data.get('rules_extracted', []))
            
            # Set default status to 'inactive' if not provided
            if 'status' not in policy_data:
                policy_data['status'] = 'inactive'
                logger.info("Status not provided, defaulting to 'inactive'")
            
            # Validate status value
            if policy_data['status'] not in ['active', 'inactive']:
                logger.warning(f"Invalid status '{policy_data['status']}', defaulting to 'inactive'")
                policy_data['status'] = 'inactive'
            
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
                logger.info(f"Created new policy: {policy_data['company']} - {policy_data['policy_name']} (status: {policy_data['status']})")
            else:
                logger.info(f"Updated existing policy: {policy_data['company']} - {policy_data['policy_name']} (status: {policy_data['status']})")
            
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
                # Get most recent active policy using status field
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
        """
        try:
            current_time = datetime.now(timezone.utc)
            
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
        ✅ UPDATED: Uses status field for active policy detection
        """
        try:
            # Get policy stats
            policy = self.get_policy(company)
            policy_stats = {
                "has_active_policy": policy is not None and policy.get('status') == 'active',  # ✅ UPDATED
                "total_rules": policy.get('total_rules', 0) if policy else 0,
                "categories": policy.get('categories', []) if policy else [],
                "policy_name": policy.get('policy_name', 'N/A') if policy else 'N/A',
                "status": policy.get('status', 'inactive') if policy else 'inactive'  # ✅ UPDATED
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
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
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
        ✅ UPDATED: Returns status field instead of effective dates
        """
        try:
            policies = list(
                self.policy_rules.find(
                    {"company": company},
                    {
                        "policy_name": 1,
                        "time_uploaded": 1,
                        "status": 1,  # ✅ UPDATED: Return status instead of dates
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
        ✅ UPDATED: Returns status field instead of effective dates
        """
        try:
            # Use policy_rules collection instead of policies
            policies = list(self.policy_rules.find(
                {"company": company},
                {
                    "_id": 1,
                    "policy_name": 1,
                    "description": 1,
                    "status": 1,  # ✅ UPDATED: Fetch status instead of dates
                    "categories": 1,
                    "total_rules": 1,
                    "last_updated": 1,
                    "time_uploaded": 1
                }
            ))
            
            # Format the response
            formatted_policies = []
            for policy in policies:
                formatted_policy = {
                    "policy_name": policy.get("policy_name"),
                    "description": policy.get("description", ""),
                    "status": policy.get("status", "inactive"),  # ✅ UPDATED: Use status field
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
        """
        Fetch a specific policy by company and policy name.
        """
        try:
            collection = self.db["policy_rules"]
            policy = collection.find_one({"company": company, "policy_name": policy_name})
            return policy
        except Exception as e:
            logging.error(f"Error fetching policy by name: {e}")
            return None
        
    def update_policy(self, company: str, policy_name: str, updated_fields: dict):
        """
        Update a policy with new fields.
        ✅ UPDATED: Validates status field if provided
        """
        try:
            # ✅ UPDATED: Validate status if being updated
            if 'status' in updated_fields:
                if updated_fields['status'] not in ['active', 'inactive']:
                    logger.warning(f"Invalid status value: {updated_fields['status']}, defaulting to 'inactive'")
                    updated_fields['status'] = 'inactive'
            
            result = self.db["policy_rules"].update_one(
                {"company": company, "policy_name": policy_name},
                {"$set": updated_fields}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated policy {policy_name} for company {company}")
            
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error updating policy: {e}")
            return False

    def get_expense_by_id(self, expense_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an expense from manual_expenses collection by ID.
        FIXED: Properly serializes all nested ObjectIds in items.
        
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
            
            # ✅ FIX: Process items to convert ObjectIds to strings
            items = []
            for item in expense.get("items", []):
                processed_item = {
                    "name": item.get("name", ""),
                    "amount": item.get("amount", 0),
                    "category": item.get("category", "Other"),
                    "quantity": item.get("quantity", 1),  # Add if exists
                }
                # Convert item _id if it exists
                if "_id" in item:
                    processed_item["_id"] = str(item["_id"])
                items.append(processed_item)
            
            # ✅ FIX: Convert ObjectId fields in breakdown if present
            breakdown = []
            for b in expense.get("breakdown", []):
                breakdown_item = {
                    "amount": b.get("amount", 0),
                    "originalAmount": b.get("originalAmount", 0),
                }
                if "category" in b and isinstance(b["category"], ObjectId):
                    breakdown_item["category"] = str(b["category"])
                breakdown.append(breakdown_item)
            
            # ✅ FIX: Handle customer ObjectId
            customer_id = expense.get("customer")
            if isinstance(customer_id, ObjectId):
                customer_id = str(customer_id)
            
            # ✅ FIX: Handle pdfBase64Data ObjectId
            pdf_data_id = expense.get("pdfBase64Data")
            if isinstance(pdf_data_id, ObjectId):
                pdf_data_id = str(pdf_data_id)
            
            # Build expense data with proper serialization
            expense_data = {
                "_id": str(expense.get("_id")),
                "title": expense.get("title", "Unknown Vendor"),
                "date": expense.get("date"),  # datetime object
                "currency": expense.get("currency", "INR"),
                "originalAmount": expense.get("originalAmount", 0),
                "totalAmount": expense.get("totalAmount", 0),
                "convertedCurrency": expense.get("convertedCurrency"),
                "convertedAmount": expense.get("convertedAmount"),
                "items": items,  # ✅ Processed items
                "receiptId": expense.get("receiptId"),
                "retailer": expense.get("retailer") or expense.get("title", "Unknown Vendor"),
                "time": expense.get("time"),
                "fileUrl": expense.get("fileUrl"),
                "status": expense.get("status", "pending"),
                "numberOfItems": expense.get("numberOfItems", len(items)),
                "breakdown": breakdown,  # ✅ Processed breakdown
                "customer": customer_id,
                "pdfBase64Data": pdf_data_id,
                "paymentMode": expense.get("paymentMode", "N/A"),  # Optional field
                "origin": expense.get("origin"),  # Optional field
                "destination": expense.get("destination"),  # Optional field,
            }
            
            logger.info(
                f"✅ Retrieved expense: {expense_data.get('title')} "
                f"with {len(items)} items, total: {expense_data.get('totalAmount')} "
                f"{expense_data.get('currency')}"
            )
            return expense_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving expense by ID: {e}", exc_info=True)
            return None
        
    def get_compliance_by_expense_id(self, expense_id: str):
        """Fetch compliance result if already exists for this expense"""
        return self.compliance_checks.find_one({"expense_id": expense_id})
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")