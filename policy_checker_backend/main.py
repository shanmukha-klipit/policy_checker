# main_url_optimized.py - FastAPI Backend with URL Support for Bills
# ✅ FIXED: Expense JSON → Bill Facts Conversion (Direct mapping, no text parsing)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any, Union
import tempfile
import os
import time
import gc
from datetime import datetime
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

# Import optimized modules with enhanced PDF extractor
from services.pdf_extractor import EnhancedPDFExtractor
from services.policy_parser import PolicyParser
from services.bill_parser import BillParser
from services.rag_engine import RAGEngine
from services.compliance_checker import ComplianceChecker
from database.mongodb_client import MongoDBClient

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", 8001))
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Policy Compliance Checker API - URL Support",
    description="RAG-powered compliance verification with Google Drive & S3 support",
    version="1.2.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if ENVIRONMENT == "development" else None,
)

# CORS configuration
ALLOWED_ORIGINS = ["*"]

if ENVIRONMENT == "production":
    production_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    ALLOWED_ORIGINS.extend(
        [origin.strip() for origin in production_origins if origin.strip()]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"⚡ {request.method} {request.url.path} "
        f"completed in {process_time:.2f}s with status {response.status_code}"
    )

    gc.collect()
    return response

# Initialize services
pdf_extractor = None
policy_parser = None
bill_parser = None
rag_engine = None
compliance_checker = None
db_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global pdf_extractor, policy_parser, bill_parser, rag_engine, compliance_checker, db_client

    logger.info(f"🚀 Starting URL-ENABLED application in {ENVIRONMENT} mode...")

    try:
        # Initialize services
        pdf_extractor = EnhancedPDFExtractor()  # Enhanced with URL support
        policy_parser = PolicyParser()
        bill_parser = BillParser()
        rag_engine = RAGEngine()
        compliance_checker = ComplianceChecker()
        db_client = MongoDBClient()

        # Warm up ML models
        logger.info("Warming up ML models...")
        _ = rag_engine.generate_embedding("warmup text")

        logger.info("✅ URL-ENABLED application started successfully")
        logger.info("📁 Supported sources: File Upload, Google Drive, AWS S3, Direct URLs")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    if db_client:
        db_client.close()
    logger.info("Application shutdown complete")

# Pydantic Models
class BillCheckURLRequest(BaseModel):
    """Request model for URL-based bill checking"""
    url: str
    company: str
    policy_name: Optional[str] = None

class URLValidationResponse(BaseModel):
    """Response model for URL validation"""
    valid: bool
    source: str
    accessible: bool
    error: Optional[str] = None
    
class BillCheckExpenseRequest(BaseModel):
    company: str
    expense_id: str
    policy_name: Optional[str] = None

# ============================================================================
# ✅ FIXED: Expense JSON → Bill Facts Conversion Helpers
# ============================================================================

def format_date(date_value) -> str:
    """
    Convert date to YYYY-MM-DD format.
    Handles datetime objects, ISO strings, and various formats.
    """
    if isinstance(date_value, datetime):
        return date_value.strftime("%Y-%m-%d")
    elif isinstance(date_value, str):
        try:
            dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d")
        except:
            if len(date_value) == 10 and date_value[4] == '-':
                return date_value
            return str(date_value)
    return str(date_value)


def build_readable_summary(expense: Dict[str, Any]) -> str:
    """
    Build a readable text summary of expense for LLM context.
    This is used as 'raw_text' for bill_facts.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"EXPENSE: {expense.get('title', 'N/A')}")
    lines.append("=" * 60)
    
    if expense.get("retailer"):
        lines.append(f"Vendor/Retailer: {expense['retailer']}")
    
    date_str = format_date(expense.get("date"))
    lines.append(f"Date: {date_str}")
    
    if expense.get("receiptId"):
        lines.append(f"Receipt ID: {expense['receiptId']}")
    
    currency = expense.get("currency", "INR")
    lines.append(f"Currency: {currency}")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append("ITEMS")
    lines.append("-" * 60)
    
    items = expense.get("items", [])
    total_items_amount = 0
    
    for idx, item in enumerate(items, 1):
        name = item.get("name", f"Item {idx}")
        amount = item.get("amount")
        category = item.get("category", "General")
        quantity = item.get("quantity", 1)
        
        lines.append(f"Item {idx}: {name}")
        lines.append(f"  Category: {category}")
        if quantity and quantity != 1:
            lines.append(f"  Quantity: {quantity}")
        if amount:
            lines.append(f"  Amount: {amount} {currency}")
            total_items_amount += amount
        lines.append("")
    
    lines.append("-" * 60)
    lines.append("SUMMARY")
    lines.append("-" * 60)
    
    original_amt = expense.get("originalAmount")
    total_amt = expense.get("totalAmount", total_items_amount)
    
    if original_amt:
        lines.append(f"Original Amount: {original_amt} {currency}")
    
    lines.append(f"Final Amount: {total_amt} {currency}")
    
    if expense.get("convertedCurrency"):
        lines.append(f"Converted Currency: {expense.get('convertedCurrency')}")
    
    if expense.get("status"):
        lines.append(f"Status: {expense['status']}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def convert_expense_to_bill_facts(expense: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ MAIN FIX: Convert expense document DIRECTLY to bill_facts structure.
    
    This bypasses the text parsing step entirely and preserves all structured data.
    This is the recommended approach for expense → bill_facts conversion.
    
    Args:
        expense: Dict with expense data from manual_expenses collection
    
    Returns:
        Dict: bill_facts structure compatible with compliance_checker
    """
    
    items = expense.get("items", [])
    
    # Calculate totals from items
    total_amount = expense.get("totalAmount", 0)
    original_amount = expense.get("originalAmount", total_amount)
    
    # Extract item information
    item_categories = set()
    item_descriptions = []
    items_list = []
    
    for item in items:
        # Collect category info
        if item.get("category"):
            item_categories.add(item["category"])
        
        # Collect item names
        if item.get("name"):
            item_descriptions.append(item["name"])
        
        # Build item detail
        items_list.append({
            "name": item.get("name", ""),
            "amount": item.get("amount", 0),
            "category": item.get("category", "Other"),
            "quantity": item.get("quantity", 1)
        })
    
    # Determine primary category (use most common or default to "Other")
    primary_category = "Other"
    if item_categories:
        # Count items by category
        category_counts = {}
        for item in items:
            cat = item.get("category", "Other")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Use most frequent category
        if category_counts:
            primary_category = max(category_counts, key=category_counts.get)
    
    logger.info(f"Primary category determined: {primary_category}")
    
    # Format date
    date_value = expense.get("date")
    date_str = format_date(date_value)
    
    # Build bill_facts with FULL STRUCTURE (matching bill_parser output)
    bill_facts = {
        # Bill identification
        "bill_id": expense.get("receiptId") or str(expense.get("_id", "unknown")),
        "source": "manual_expense",
        "expense_id": str(expense.get("_id", "")),
        
        # Main bill metadata
        "bill_meta": {
            # Category & Amount
            "category": primary_category,
            "amount": float(total_amount) if total_amount else 0,
            "currency": expense.get("currency", "INR"),
            "original_amount": float(original_amount) if original_amount else float(total_amount) if total_amount else 0,
            
            # Vendor & Description
            "vendor": expense.get("retailer") or expense.get("title", "Unknown Vendor"),
            "description": " | ".join(item_descriptions) if item_descriptions else expense.get("title", ""),
            
            # Date & Time
            "date": date_str,  # YYYY-MM-DD format
            "receipt_date": date_str,
            
            # Metadata
            "mode": expense.get("paymentMode", "N/A"),
            "origin": expense.get("origin", "N/A"),
            "destination": expense.get("destination", "N/A"),
            "invoice_number": expense.get("receiptId", ""),
            "has_receipt": bool(expense.get("receiptId")),
            "approval_status": expense.get("status", "pending"),
            
            # Item-level details (NEW: structured items)
            "items": items_list,
            "item_count": len(items),
            "total_items": len(items),
            
            # Conversion details
            "converted_currency": expense.get("convertedCurrency"),
            "converted_amount": expense.get("convertedAmount"),
        },
        
        # Raw text for LLM context
        "raw_text": build_readable_summary(expense),
        
        # Processing metadata
        "parsed_at": datetime.utcnow().isoformat(),
        "confidence": 0.95,  # High confidence since data is already structured
        "parsing_method": "direct_json_conversion",  # Track method
    }
    
    logger.info(
        f"✅ Converted expense to bill_facts: "
        f"id={bill_facts['bill_id']}, "
        f"category={primary_category}, "
        f"amount={total_amount}, "
        f"items={len(items)}"
    )
    
    return bill_facts

# ============================================================================
# END: Helper Functions
# ============================================================================

# Helper Functions
async def validate_file_size(file: UploadFile) -> int:
    """Validate uploaded file size"""
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB",
        )
    return file_size

# Routes
@app.get("/")
async def root():
    return {
        "message": "Policy Compliance Checker API - URL Support",
        "version": "1.2.0",
        "status": "running",
        "environment": ENVIRONMENT,
        "features": [
            "File uploads",
            "Google Drive URLs",
            "AWS S3 URLs",
            "Direct HTTP/HTTPS URLs",
            "Batch LLM processing (80% faster)",
            "✅ Direct JSON expense conversion (NEW)"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        companies = db_client.list_companies()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        db_status = "disconnected"

    # Check PDF extractor capabilities
    pdf_capabilities = {
        "google_drive": pdf_extractor.drive_service is not None,
        "aws_s3": pdf_extractor.s3_client is not None,
        "direct_urls": True
    }

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "environment": ENVIRONMENT,
        "database": db_status,
        "pdf_sources": pdf_capabilities,
        "timestamp": datetime.utcnow().isoformat(),
        "optimized": True
    }

@app.post("/api/policy/upload")
async def upload_policy(
    file: UploadFile = File(...),
    company: str = Form(...),
    policy_name: str = Form(...),
    effective_from: Optional[str] = Form(None),
    effective_to: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """Upload and process policy (same as before)"""
    try:
        logger.info(f"⚡ Processing policy upload for company: {company}")

        # ✅ Default effective_from to today's date if not provided
        if not effective_from:
            effective_from = datetime.utcnow().strftime("%Y-%m-%d")

        file_size = await validate_file_size(file)
        logger.info(f"File size: {file_size/1024:.2f}KB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        try:
            policy_text = pdf_extractor.extract_text(temp_path)

            if not policy_text or len(policy_text.strip()) < 100:
                raise HTTPException(
                    status_code=400, detail="Could not extract sufficient text from PDF"
                )

            parsed_data = policy_parser.parse_policy(policy_text, company)
            rules = parsed_data["rules"]
            categories = parsed_data["categories"]

            if not rules:
                raise HTTPException(
                    status_code=400,
                    detail="No rules could be extracted from the policy",
                )

            logger.info(f"Generating embeddings for {len(rules)} rules...")
            for rule in rules:
                rule["embedding"] = rag_engine.generate_embedding(rule["raw_text"])

            policy_data = {
                "company": company,
                "file_path": file.filename,
                "policy_name": policy_name,
                "description": description or "",
                "rules_extracted": rules,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "categories": categories,
                "embeddings_model": "models/text-embedding-004",
                "total_rules": len(rules),
            }

            success = db_client.store_policy(policy_data)

            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to store policy in database"
                )

            logger.info(f"✅ Successfully stored policy with {len(rules)} rules for {company}")

            return JSONResponse(
                {
                    "status": "success",
                    "company": company,
                    "policy_name": policy_name,
                    "description": description,
                    "rules_count": len(rules),
                    "rules_extracted": rules,
                    "categories": categories,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "message": "Policy uploaded and indexed successfully",
                }
            )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing policy: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/bill/check/expense")
async def check_bill_from_expense(request: BillCheckExpenseRequest):
    """
    ✅ FIXED: Check bill compliance from manual expense ID.
    
    Uses direct JSON-to-bill_facts conversion (no text parsing).
    This preserves all structured data and maintains consistency with PDF uploads.
    """
    try:
        logger.info(
            f"⚡ Checking bill from expense ID for {request.company}, "
            f"expense_id={request.expense_id}, policy_name={request.policy_name or 'ALL'}"
        )

        # Fetch expense from manual_expenses collection
        expense = db_client.get_expense_by_id(request.expense_id)
        
        if not expense:
            raise HTTPException(
                status_code=404,
                detail=f"Expense not found with ID: {request.expense_id}"
            )
        
        logger.info(f"📄 Found expense: {expense.get('title', 'N/A')}")

        # ✅ KEY FIX: Direct conversion instead of text extraction
        bill_facts = convert_expense_to_bill_facts(expense)
        
        logger.info(f"📋 Bill facts structure: {bill_facts.get('bill_id')}, "
                    f"category={bill_facts['bill_meta'].get('category')}, "
                    f"amount={bill_facts['bill_meta'].get('amount')}, "
                    f"items={bill_facts['bill_meta'].get('item_count')}")
        
        # Ensure correct date for compliance checking
        bill_facts['bill_meta']['date'] = format_date(expense.get("date"))
        logger.info(f"📅 Bill date for compliance check: {bill_facts['bill_meta']['date']}")

        # Get stored categories
        stored_categories = db_client.get_allowed_categories(request.company)
        if not stored_categories:
            raise HTTPException(
                status_code=404,
                detail=f"No policy found for company: {request.company}. Please upload policy first.",
            )

        # Run compliance check (optimized batch processing)
        compliance_result = compliance_checker.check_compliance(
            bill_facts=bill_facts,
            company=request.company,
            stored_categories=stored_categories,
            policy_name=request.policy_name,
        )

        # Generate report
        report = compliance_checker.generate_detailed_report(compliance_result)

        # Store compliance check
        db_client.store_compliance_check(
            {
                "company": request.company,
                "file_path": expense.get("fileUrl"),
                "source": "manual_expense",
                "expense_id": request.expense_id,
                "violations": report["violations"],
                "classification": {
                    "compliant_items": 1 if report["is_compliant"] else 0,
                    "non_compliant_items": 0 if report["is_compliant"] else 1,
                    "total_items": 1,
                    "compliance_score": report["compliance_score"],
                    "categories_checked": stored_categories,
                },
                "bill_id": bill_facts.get("bill_id"),
                "policy_name": request.policy_name,
                "metadata": {
                    "bill_category": report["category_info"]["bill_category"],
                    "matched_category": report["category_info"]["matched_category"],
                    "similarity": report["category_info"]["similarity"],
                    "source_type": "manual_expense",
                    "expense_title": expense.get("title"),
                    "expense_date": expense.get("date"),
                    "currency": expense.get("currency"),
                    "original_amount": expense.get("originalAmount"),
                    "item_count": len(expense.get("items", [])),
                },
            }
        )

        logger.info(f"✅ Compliance check completed. Score: {report['compliance_score']}")

        # Convert datetime to ISO string for JSON serialization
        expense_date = expense.get("date")
        if isinstance(expense_date, datetime):
            expense_date = expense_date.isoformat()
        
        return JSONResponse(
            {
                "company": request.company,
                "expense_id": request.expense_id,
                "bill_id": bill_facts.get("bill_id"),
                "source": "manual_expense",
                "expense_title": expense.get("title"),
                "expense_date": expense_date,
                "currency": expense.get("currency"),
                "original_amount": expense.get("originalAmount"),
                "total_amount": expense.get("totalAmount"),
                "item_count": len(expense.get("items", [])),
                "policy_name": request.policy_name,
                "overall_score": report["compliance_score"],
                "is_compliant": report["is_compliant"],
                "category_info": report["category_info"],
                "mismatch_count": report["total_violations"],
                "severity_breakdown": report["severity_breakdown"],
                "mismatches": report["violations"],
                "summary": report["summary"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking bill from expense: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/bill/check/url")
async def check_bill_from_url(request: BillCheckURLRequest):
    """
    🆕 NEW: Check bill compliance from URL (Google Drive, S3, or direct URL).
    
    Supports:
    - Google Drive: https://drive.google.com/file/d/FILE_ID/view
    - AWS S3: https://bucket.s3.region.amazonaws.com/key or s3://bucket/key
    - Direct URLs: https://example.com/bill.pdf
    """
    try:
        logger.info(
            f"⚡ Checking bill from URL for {request.company}, "
            f"policy_name={request.policy_name or 'ALL'}"
        )
        logger.info(f"📄 URL: {request.url}")

        # Validate URL first
        url_validation = pdf_extractor.validate_url(request.url)
        
        if not url_validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid URL: {url_validation.get('error', 'Unknown error')}"
            )
        
        if not url_validation["accessible"]:
            raise HTTPException(
                status_code=400,
                detail=f"URL not accessible: {url_validation.get('error', 'Check credentials')}"
            )
        
        logger.info(f"📁 Source: {url_validation['source']}")

        # Extract text from URL
        try:
            bill_text = pdf_extractor.extract_from_url(request.url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not bill_text or len(bill_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from PDF at URL",
            )

        # Parse bill
        bill_facts = bill_parser.parse_bill(bill_text)

        # Get stored categories
        stored_categories = db_client.get_allowed_categories(request.company)
        if not stored_categories:
            raise HTTPException(
                status_code=404,
                detail=f"No policy found for company: {request.company}. Please upload policy first.",
            )

        # Run compliance check (optimized batch processing)
        compliance_result = compliance_checker.check_compliance(
            bill_facts=bill_facts,
            company=request.company,
            stored_categories=stored_categories,
            policy_name=request.policy_name,
        )

        # Generate report
        report = compliance_checker.generate_detailed_report(compliance_result)

        # Store compliance check
        db_client.store_compliance_check(
            {
                "company": request.company,
                "file_path": request.url,  # Store URL instead of filename
                "source": url_validation["source"],
                "violations": report["violations"],
                "classification": {
                    "compliant_items": 1 if report["is_compliant"] else 0,
                    "non_compliant_items": 0 if report["is_compliant"] else 1,
                    "total_items": 1,
                    "compliance_score": report["compliance_score"],
                    "categories_checked": stored_categories,
                },
                "bill_id": bill_facts.get("bill_id"),
                "policy_name": request.policy_name,
                "metadata": {
                    "bill_category": report["category_info"]["bill_category"],
                    "matched_category": report["category_info"]["matched_category"],
                    "similarity": report["category_info"]["similarity"],
                    "source_type": url_validation["source"],
                },
            }
        )

        logger.info(f"✅ Compliance check completed. Score: {report['compliance_score']}")

        return JSONResponse(
            {
                "company": request.company,
                "bill_id": bill_facts.get("bill_id"),
                "source": url_validation["source"],
                "url": request.url,
                "policy_name": request.policy_name,
                "overall_score": report["compliance_score"],
                "is_compliant": report["is_compliant"],
                "category_info": report["category_info"],
                "mismatch_count": report["total_violations"],
                "severity_breakdown": report["severity_breakdown"],
                "mismatches": report["violations"],
                "summary": report["summary"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking bill from URL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/bill/check")
async def check_bill_from_file(
    file: UploadFile = File(...),
    company: str = Form(...),
    policy_name: Optional[str] = Form(None),
):
    """
    Check bill compliance from uploaded file (backward compatibility).
    For URL-based checking, use /api/bill/check/url instead.
    """
    try:
        logger.info(
            f"⚡ Checking uploaded bill for {company}, policy_name={policy_name or 'ALL'}"
        )

        file_size = await validate_file_size(file)
        logger.info(f"File size: {file_size/1024:.2f}KB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        try:
            bill_text = pdf_extractor.extract_text(temp_path)
            
            if not bill_text or len(bill_text.strip()) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract sufficient text from bill",
                )

            bill_facts = bill_parser.parse_bill(bill_text)

            stored_categories = db_client.get_allowed_categories(company)
            if not stored_categories:
                raise HTTPException(
                    status_code=404,
                    detail=f"No policy found for company: {company}. Please upload policy first.",
                )

            compliance_result = compliance_checker.check_compliance(
                bill_facts=bill_facts,
                company=company,
                stored_categories=stored_categories,
                policy_name=policy_name,
            )

            report = compliance_checker.generate_detailed_report(compliance_result)

            db_client.store_compliance_check(
                {
                    "company": company,
                    "file_path": file.filename,
                    "source": "file_upload",
                    "violations": report["violations"],
                    "classification": {
                        "compliant_items": 1 if report["is_compliant"] else 0,
                        "non_compliant_items": 0 if report["is_compliant"] else 1,
                        "total_items": 1,
                        "compliance_score": report["compliance_score"],
                        "categories_checked": stored_categories,
                    },
                    "bill_id": bill_facts.get("bill_id"),
                    "policy_name": policy_name,
                    "metadata": {
                        "bill_category": report["category_info"]["bill_category"],
                        "matched_category": report["category_info"]["matched_category"],
                        "similarity": report["category_info"]["similarity"],
                    },
                }
            )

            logger.info(f"✅ Compliance check completed. Score: {report['compliance_score']}")

            return JSONResponse(
                {
                    "company": company,
                    "bill_id": bill_facts.get("bill_id"),
                    "source": "file_upload",
                    "policy_name": policy_name,
                    "overall_score": report["compliance_score"],
                    "is_compliant": report["is_compliant"],
                    "category_info": report["category_info"],
                    "mismatch_count": report["total_violations"],
                    "severity_breakdown": report["severity_breakdown"],
                    "mismatches": report["violations"],
                    "summary": report["summary"],
                }
            )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking bill: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/bill/validate-url")
async def validate_bill_url(url: str = Form(...)):
    """
    🆕 NEW: Validate if a URL is accessible and supported.
    Use this to check URLs before attempting to process them.
    """
    try:
        validation = pdf_extractor.validate_url(url)
        
        return JSONResponse(
            {
                "url": url,
                "valid": validation["valid"],
                "source": validation["source"],
                "accessible": validation["accessible"],
                "error": validation.get("error"),
                "supported_sources": [
                    "google_drive",
                    "aws_s3",
                    "direct_url"
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error validating URL: {e}")
        return JSONResponse(
            {
                "url": url,
                "valid": False,
                "error": str(e)
            },
            status_code=400
        )

# All other existing endpoints remain unchanged
@app.get("/api/policy/{company}/rules")
async def get_policy_rules(company: str):
    """Get all rules for a company's policy"""
    try:
        rules = db_client.get_policy_rules(company)
        if not rules:
            raise HTTPException(
                status_code=404, detail=f"No policy found for company: {company}"
            )
        return JSONResponse({"company": company, "rules": rules})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching rules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies")
async def list_companies():
    """List all companies with policies"""
    try:
        companies = db_client.list_companies()
        return JSONResponse({"total": len(companies), "companies": companies})
    except Exception as e:
        logger.error(f"Error listing companies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/policy/{company}")
async def delete_policy(company: str):
    """Delete policy for a company"""
    try:
        success = db_client.delete_policy(company)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"No policy found for company: {company}"
            )
        return JSONResponse(
            {"status": "success", "message": f"Policy deleted for {company}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting policy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/policies/{company}")
async def list_company_policies(company: str):
    """
    Get all policies for a company (name, description, status, effective dates, etc.)
    """
    try:
        policies = db_client.get_policies_by_company(company)

        if not policies:         
            return JSONResponse(
                {
                    "company": company,
                    "total_policies": 0,
                    "policies": []
                }
                        
            )

        # Return summarized metadata, not full rules
        response = []
        current_date = datetime.now().date()
        
        for p in policies:
            # Dynamically determine status based on dates
            effective_from = p.get("effective_from")
            effective_to = p.get("effective_to")
            
            status = "active"  # default
            
            if effective_from and effective_to:
                try:
                    # Parse dates if they're strings
                    if isinstance(effective_from, str):
                        effective_from = datetime.fromisoformat(effective_from.replace('Z', '+00:00')).date()
                    elif isinstance(effective_from, datetime):
                        effective_from = effective_from.date()
                    
                    if isinstance(effective_to, str):
                        effective_to = datetime.fromisoformat(effective_to.replace('Z', '+00:00')).date()
                    elif isinstance(effective_to, datetime):
                        effective_to = effective_to.date()
                    
                    # Determine status
                    if current_date < effective_from:
                        status = "inactive"  # or "pending" if you prefer
                    elif current_date > effective_to:
                        status = "inactive"  # or "expired"
                    else:
                        status = "active"
                except Exception as date_parse_error:
                    logger.warning(f"Error parsing dates for policy {p.get('policy_name')}: {date_parse_error}")
                    status = p.get("status", "active")  # fallback to stored status
            else:
                status = p.get("status", "active")  # fallback if dates not present
            
            response.append(
                {
                    "policy_name": p.get("policy_name"),
                    "description": p.get("description", ""),
                    "status": status,
                    "effective_from": p.get("effective_from"),
                    "effective_to": p.get("effective_to"),
                    "total_rules": p.get("total_rules", 0),
                    "categories": p.get("categories", []),
                    "last_updated": p.get(
                        "updated_at",
                        (
                            p.get("_id").generation_time.isoformat()
                            if "_id" in p
                            else None
                        ),
                    ),
                }
            )

        return JSONResponse(
            {"company": company, "total_policies": len(response), "policies": response}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching policies for {company}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    
@app.get("/api/policy/{company}/{policy_name}")
async def get_policy_by_name(company: str, policy_name: str):
    """Get a specific policy for a company"""
    try:
        policy = db_client.get_policy_by_name(company, policy_name)
        if not policy:
            raise HTTPException(
                status_code=404,
                detail=f"Policy '{policy_name}' not found for company '{company}'"
            )

        # Dynamically determine status based on dates
        current_date = datetime.now().date()
        effective_from = policy.get("effective_from")
        effective_to = policy.get("effective_to")
        
        status = "active"  # default
        
        if effective_from and effective_to:
            try:
                # Parse dates if they're strings
                if isinstance(effective_from, str):
                    effective_from = datetime.fromisoformat(effective_from.replace('Z', '+00:00')).date()
                elif isinstance(effective_from, datetime):
                    effective_from = effective_from.date()
                
                if isinstance(effective_to, str):
                    effective_to = datetime.fromisoformat(effective_to.replace('Z', '+00:00')).date()
                elif isinstance(effective_to, datetime):
                    effective_to = effective_to.date()
                
                # Determine status
                if current_date < effective_from:
                    status = "inactive"  # or "pending"
                elif current_date > effective_to:
                    status = "inactive"  # or "expired"
                else:
                    status = "active"
            except Exception as date_parse_error:
                logger.warning(f"Error parsing dates for policy {policy_name}: {date_parse_error}")
                status = policy.get("status", "active")  # fallback to stored status
        else:
            status = policy.get("status", "active")  # fallback if dates not present

        return JSONResponse({
            "company": company,
            "policy_name": policy.get("policy_name"),
            "description": policy.get("description", ""),
            "status": status,
            "effective_from": policy.get("effective_from"),
            "effective_to": policy.get("effective_to"),
            "total_rules": policy.get("total_rules", 0),
            "categories": policy.get("categories", []),
            "rules_extracted": policy.get("rules_extracted", []),
            "embeddings_model": policy.get("embeddings_model"),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching policy '{policy_name}' for {company}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    
@app.put("/api/policy/{company}/{policy_name}")
async def update_policy(
    company: str,
    policy_name: str,
    file: Optional[UploadFile] = File(None),
    description: Optional[str] = Form(None),
    effective_from: Optional[str] = Form(None),
    effective_to: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
):
    """
    Update an existing policy:
    - If JSON fields are given (description, effective dates, status): only update those.
    - If file is provided: re-extract rules, categories, and embeddings (like upload),
      and replace those fields in DB.
    Returns full updated policy document.
    """
    try:
        logger.info(f"Updating policy '{policy_name}' for company '{company}'")

        existing_policy = db_client.get_policy_by_name(company, policy_name)
        if not existing_policy:
            raise HTTPException(
                status_code=404,
                detail=f"Policy '{policy_name}' not found for company '{company}'"
            )

        updated_fields = {}
        if description is not None:
            updated_fields["description"] = description
        if effective_from is not None:
            updated_fields["effective_from"] = effective_from
        if effective_to is not None:
            updated_fields["effective_to"] = effective_to
        if status is not None:
            updated_fields["status"] = status

        # 🔄 File reprocessing
        if file:
            logger.info(f"Re-uploading file for policy '{policy_name}' to re-extract rules")
            file_size = await validate_file_size(file)
            logger.info(f"File size: {file_size/1024:.2f}KB")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_path = tmp_file.name

            try:
                policy_text = pdf_extractor.extract_text(temp_path)
                if not policy_text or len(policy_text.strip()) < 100:
                    raise HTTPException(status_code=400, detail="Could not extract sufficient text from PDF")

                parsed_data = policy_parser.parse_policy(policy_text, company)
                rules = parsed_data["rules"]
                categories = parsed_data["categories"]

                if not rules:
                    raise HTTPException(status_code=400, detail="No rules could be extracted from the uploaded file")

                for rule in rules:
                    rule["embedding"] = rag_engine.generate_embedding(rule["raw_text"])

                updated_fields.update({
                    "file_path": file.filename,
                    "rules_extracted": rules,
                    "categories": categories,
                    "total_rules": len(rules),
                    "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2"
                })

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        updated_fields["updated_at"] = datetime.utcnow().isoformat()

        success = db_client.update_policy(company, policy_name, updated_fields)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update policy in database")

        updated_policy = db_client.get_policy_by_name(company, policy_name)

        # 🧠 Compute status dynamically (same as get_policy_by_name)
        current_date = datetime.now().date()
        effective_from = updated_policy.get("effective_from")
        effective_to = updated_policy.get("effective_to")

        computed_status = "active"
        try:
            if effective_from and effective_to:
                if isinstance(effective_from, str):
                    effective_from = datetime.fromisoformat(effective_from.replace('Z', '+00:00')).date()
                elif isinstance(effective_from, datetime):
                    effective_from = effective_from.date()

                if isinstance(effective_to, str):
                    effective_to = datetime.fromisoformat(effective_to.replace('Z', '+00:00')).date()
                elif isinstance(effective_to, datetime):
                    effective_to = effective_to.date()

                if current_date < effective_from:
                    computed_status = "inactive"
                elif current_date > effective_to:
                    computed_status = "inactive"
                else:
                    computed_status = "active"
            else:
                computed_status = updated_policy.get("status", "active")
        except Exception as e:
            logger.warning(f"Error parsing dates while computing status for {policy_name}: {e}")
            computed_status = updated_policy.get("status", "active")

        # ✅ Return in SAME format as get_policy_by_name
        return JSONResponse({
            "company": company,
            "policy_name": updated_policy.get("policy_name"),
            "description": updated_policy.get("description", ""),
            "status": computed_status,
            "effective_from": updated_policy.get("effective_from"),
            "effective_to": updated_policy.get("effective_to"),
            "total_rules": updated_policy.get("total_rules", 0),
            "categories": updated_policy.get("categories", []),
            "rules_extracted": updated_policy.get("rules_extracted", []),
            "embeddings_model": updated_policy.get("embeddings_model"),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating policy '{policy_name}' for {company}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting URL-ENABLED FastAPI application")
    uvicorn.run(
        app, host="0.0.0.0", port=PORT, log_level=log_level.lower(), access_log=True
    )