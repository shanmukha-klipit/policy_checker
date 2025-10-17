# main.py - FastAPI Backend for Policy Compliance Checker (Production Ready)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import time
import gc
from datetime import datetime
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

# Import custom modules
from services.pdf_extractor import PDFExtractor
from services.policy_parser import PolicyParser
from services.bill_parser import BillParser
from services.rag_engine import RAGEngine
from services.compliance_checker import ComplianceChecker
from database.mongodb_client import MongoDBClient

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", 8001))
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10MB default

# Setup logging for production
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Policy Compliance Checker API",
    description="RAG-powered compliance verification system",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,  # Disable docs in production
    redoc_url="/redoc" if ENVIRONMENT == "development" else None
)

# CORS configuration
ALLOWED_ORIGINS = ["*"]

# Add production origins
if ENVIRONMENT == "production":
    production_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    ALLOWED_ORIGINS.extend([origin.strip() for origin in production_origins if origin.strip()])

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
        f"{request.method} {request.url.path} "
        f"completed in {process_time:.2f}s with status {response.status_code}"
    )
    
    # Cleanup after request
    gc.collect()
    
    return response

# Initialize services (lazy loading in startup event)
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
    
    logger.info(f"Starting application in {ENVIRONMENT} mode...")
    
    try:
        # Initialize services
        pdf_extractor = PDFExtractor()
        policy_parser = PolicyParser()
        bill_parser = BillParser()
        rag_engine = RAGEngine()
        compliance_checker = ComplianceChecker()
        db_client = MongoDBClient()
        
        # Warm up ML models to avoid cold start
        logger.info("Warming up ML models...")
        _ = rag_engine.generate_embedding("warmup text")
        
        logger.info("Application started successfully")
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

# ----------------------------
# Pydantic Models
# ----------------------------
class PolicyRule(BaseModel):
    rule_id: str
    category: str
    attributes: Dict[str, Any]
    raw_text: str
    embedding: Optional[List[float]] = None

class BillFact(BaseModel):
    bill_id: str
    company: str
    category: str
    bill_meta: Dict[str, Any]
    raw_text: str

class Mismatch(BaseModel):
    classification: str
    severity: str
    explanation: str
    confidence: float
    company_rule_text: str
    bill_snippet: str

class ComplianceReport(BaseModel):
    company: str
    overall_score: int
    mismatch_count: int
    mismatches: List[Mismatch]

# ----------------------------
# Helper Functions
# ----------------------------
async def validate_file_size(file: UploadFile) -> int:
    """Validate uploaded file size"""
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB"
        )
    return file_size

async def validate_pdf_file(file: UploadFile):
    """Validate file is a PDF"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "Policy Compliance Checker API",
        "version": "1.0.0",
        "status": "running",
        "environment": ENVIRONMENT
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection
        companies = db_client.list_companies()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        db_status = "disconnected"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "environment": ENVIRONMENT,
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/policy/upload")
async def upload_policy(
    file: UploadFile = File(...), 
    company: str = Form(...),
    policy_name: str = Form(...),
    effective_from: str = Form(...),
    effective_to: str = Form(None),
    description: str = Form(None)
):
    """Upload and process a company policy document"""
    try:
        logger.info(f"Processing policy upload for company: {company}")
        
        # Validate file
        # await validate_pdf_file(file)
        file_size = await validate_file_size(file)
        logger.info(f"File size: {file_size/1024:.2f}KB")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        try:
            # Extract text from PDF
            policy_text = pdf_extractor.extract_text(temp_path)
            
            if not policy_text or len(policy_text.strip()) < 100:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract sufficient text from PDF"
                )
            
            # Parse policy
            parsed_data = policy_parser.parse_policy(policy_text, company)
            rules = parsed_data['rules']
            categories = parsed_data['categories']
            
            if not rules:
                raise HTTPException(
                    status_code=400,
                    detail="No rules could be extracted from the policy"
                )
            
            # Generate embeddings for rules
            logger.info(f"Generating embeddings for {len(rules)} rules...")
            for rule in rules:
                rule['embedding'] = rag_engine.generate_embedding(rule['raw_text'])
            
            # Prepare complete policy document
            policy_data = {
                "company": company,
                "file_path": file.filename,
                "policy_name": policy_name,
                "description": description if description else "",
                "rules_extracted": rules,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "categories": categories,
                "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
                "total_rules": len(rules)
            }
            
            # Store in MongoDB
            success = db_client.store_policy(policy_data)
            
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to store policy in database"
                )
            
            logger.info(f"Successfully stored policy with {len(rules)} rules for {company}")
            
            return JSONResponse({
                "status": "success",
                "company": company,
                "policy_name": policy_name,
                "description": description,
                "rules_count": len(rules),
                "rules_extracted": rules,
                "categories": categories,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "message": "Policy uploaded and indexed successfully"
            })
            
        finally:
            # Always cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing policy: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/bill/check")
async def check_bill(
    file: UploadFile = File(...), 
    company: str = Form(...)
):
    """Check bill compliance against company policy"""
    try:
        logger.info(f"Checking bill compliance for {company}")
        
        # Validate file
        # await validate_pdf_file(file)
        file_size = await validate_file_size(file)
        logger.info(f"File size: {file_size/1024:.2f}KB")

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        try:
            # Extract and parse bill
            bill_text = pdf_extractor.extract_text(temp_path)
            
            if not bill_text or len(bill_text.strip()) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract sufficient text from bill"
                )
            
            bill_facts = bill_parser.parse_bill(bill_text)
            
            # Get stored categories for the company
            stored_categories = db_client.get_allowed_categories(company)
            
            if not stored_categories:
                raise HTTPException(
                    status_code=404,
                    detail=f"No policy found for company: {company}. Please upload policy first."
                )
            
            # Run compliance check
            compliance_result = compliance_checker.check_compliance(
                bill_facts=bill_facts,
                company=company,
                stored_categories=stored_categories
            )
            
            # Generate detailed report
            report = compliance_checker.generate_detailed_report(compliance_result)
            
            # Store in database
            db_client.store_compliance_check({
                "company": company,
                "file_path": file.filename,
                "violations": report['violations'],
                "classification": {
                    "compliant_items": 1 if report['is_compliant'] else 0,
                    "non_compliant_items": 0 if report['is_compliant'] else 1,
                    "total_items": 1,
                    "compliance_score": report['compliance_score'],
                    "categories_checked": stored_categories
                },
                "bill_id": bill_facts.get('bill_id'),
                "policy_name": None,
                "metadata": {
                    "bill_category": report['category_info']['bill_category'],
                    "matched_category": report['category_info']['matched_category'],
                    "similarity": report['category_info']['similarity']
                }
            })

            logger.info(f"Compliance check completed. Score: {report['compliance_score']}")

            return JSONResponse({
                "company": company,
                "bill_id": bill_facts.get('bill_id'),
                "overall_score": report['compliance_score'],
                "is_compliant": report['is_compliant'],
                "category_info": report['category_info'],
                "mismatch_count": report['total_violations'],
                "severity_breakdown": report['severity_breakdown'],
                "mismatches": report['violations'],
                "summary": report['summary']
            })
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking bill: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/policy/{company}/rules")
async def get_policy_rules(company: str):
    """Get all rules for a company's policy"""
    try:
        rules = db_client.get_policy_rules(company)
        if not rules:
            raise HTTPException(
                status_code=404,
                detail=f"No policy found for company: {company}"
            )
        return JSONResponse({"company": company, "rules": rules})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching rules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/policy/{company}/categories")
async def get_policy_categories(company: str):
    """Get all categories for a company's policy"""
    try:
        categories = db_client.get_allowed_categories(company)
        if not categories:
            raise HTTPException(
                status_code=404,
                detail=f"No policy found for company: {company}"
            )
        return JSONResponse({"company": company, "categories": categories})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/compliance/history/{company}")
async def get_compliance_history(company: str, limit: int = 10):
    """Get compliance check history for a company"""
    try:
        history = db_client.get_compliance_history(company, limit)
        return JSONResponse({
            "company": company,
            "total": len(history),
            "history": history
        })
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics/{company}")
async def get_statistics(company: str):
    """Get comprehensive statistics for a company"""
    try:
        stats = db_client.get_statistics(company)
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for company: {company}"
            )
        return JSONResponse(stats)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies")
async def list_companies():
    """List all companies with policies"""
    try:
        companies = db_client.list_companies()
        return JSONResponse({
            "total": len(companies),
            "companies": companies
        })
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
                status_code=404,
                detail=f"No policy found for company: {company}"
            )
        return JSONResponse({
            "status": "success",
            "message": f"Policy deleted for {company}"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting policy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level=log_level.lower(),
        access_log=True
    )