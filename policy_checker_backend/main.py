# main.py - FastAPI Backend for Policy Compliance Checker

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv()

# Import custom modules
from services.pdf_extractor import PDFExtractor
from services.policy_parser import PolicyParser
from services.bill_parser import BillParser
from services.rag_engine import RAGEngine
from services.compliance_checker import ComplianceChecker
from database.mongodb_client import MongoDBClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Policy Compliance Checker API",
    description="RAG-powered compliance verification system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_extractor = PDFExtractor()
policy_parser = PolicyParser()
bill_parser = BillParser()
rag_engine = RAGEngine()
compliance_checker = ComplianceChecker()
db_client = MongoDBClient()

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
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "Policy Compliance Checker API",
        "version": "1.0.0",
        "status": "running"
    }

# @app.post("/api/policy/upload")
# async def upload_policy(file: UploadFile = File(...), company: str = Form(...)):
#     try:
#         logger.info(f"Processing policy upload for company: {company}")

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(await file.read())
#             temp_path = tmp_file.name

#         # Extract text
#         policy_text = pdf_extractor.extract_text(temp_path)
        
#         # Parse policy - NOW returns dict with 'rules' and 'categories'
#         parsed_data = policy_parser.parse_policy(policy_text, company)
        
#         rules = parsed_data['rules']
#         categories = parsed_data['categories']
        
#         # Generate embeddings for rules
#         for rule in rules:
#             rule['embedding'] = rag_engine.generate_embedding(rule['raw_text'])
        
#         # Store rules in MongoDB
#         db_client.store_policy(company, rules)
        
#         # IMPORTANT: Store categories for this company
#         # db_client.store_company_categories(company, categories)
        
#         os.remove(temp_path)

#         logger.info(f"Processed {len(rules)} rules across {len(categories)} categories for {company}")
        
#         return JSONResponse({
#             "status": "success",
#             "company": company,
#             "rules_count": len(rules),
#             "categories": categories,
#             "message": "Policy uploaded and indexed successfully"
#         })
#     except Exception as e:
#         logger.error(f"Error processing policy: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy/upload")
async def upload_policy(
    file: UploadFile = File(...), 
    company: str = Form(...),
    policy_name: str = Form(...),
    effective_from: str = Form(...),
    effective_to: str = Form(None),
    description: str = Form(None)  # NEW: Added description field
):
    try:
        logger.info(f"Processing policy upload for company: {company}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            temp_path = tmp_file.name

        # Extract text from PDF
        policy_text = pdf_extractor.extract_text(temp_path)
        
        # Parse policy - returns dict with 'rules' and 'categories'
        parsed_data = policy_parser.parse_policy(policy_text, company)
        
        rules = parsed_data['rules']
        categories = parsed_data['categories']
        
        # Generate embeddings for rules
        for rule in rules:
            rule['embedding'] = rag_engine.generate_embedding(rule['raw_text'])
        
        # Prepare complete policy document
        policy_data = {
            "company": company,
            "file_path": file.filename,  # Store original filename
            "policy_name": policy_name,
            "description": description if description else "",  # NEW: Added description
            "rules_extracted": rules,
            "effective_from": effective_from,
            "effective_to": effective_to,  # Can be None
            "categories": categories,
            "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",  # Update with your actual model
            "total_rules": len(rules)
        }
        
        # Store complete policy in MongoDB (single document)
        success = db_client.store_policy(policy_data)
        
        # Clean up temporary file
        os.remove(temp_path)

        if success:
            logger.info(f"Successfully stored policy with {len(rules)} rules across {len(categories)} categories for {company}")
            
            return JSONResponse({
                "status": "success",
                "company": company,
                "policy_name": policy_name,
                "description": description,  # NEW: Include in response
                "rules_count": len(rules),
                "categories": categories,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "message": "Policy uploaded and indexed successfully"
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to store policy in database")
            
    except Exception as e:
        logger.error(f"Error processing policy: {str(e)}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

        
@app.post("/api/bill/check")
async def check_bill(file: UploadFile = File(...), company: str = Form(...)):
    try:
        logger.info(f"Checking bill compliance for {company}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            temp_path = tmp_file.name

        bill_text = pdf_extractor.extract_text(temp_path)
        bill_facts = bill_parser.parse_bill(bill_text)
        
        # Get stored categories for the company
        stored_categories = db_client.get_allowed_categories(company)
        
        # Run compliance check with category validation
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
            "bill_id": bill_facts.get('bill_id'),
            "result": compliance_result,
            "report": report,
            "timestamp": datetime.utcnow().isoformat()
        })

        os.remove(temp_path)
        logger.info(f"Compliance check completed. Score: {report['compliance_score']}")

        return JSONResponse({
            "company": company,
            "bill_id": bill_facts.get('bill_id'),
            "overall_score": report['compliance_score'],
            "is_compliant": report['is_compliant'],
            "category_info": report['category_info'],
            "mismatch_count": report['total_violations'],
            "mismatches": report['violations'],
            "summary": report['summary']
        })
    except Exception as e:
        logger.error(f"Error checking bill: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

        
@app.get("/api/policy/{company}/rules")
async def get_policy_rules(company: str):
    try:
        rules = db_client.get_policy_rules(company)
        return JSONResponse({"company": company, "rules": rules})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/compliance/history/{company}")
async def get_compliance_history(company: str, limit: int = 10):
    try:
        history = db_client.get_compliance_history(company, limit)
        return JSONResponse({"company": company, "history": history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/policy/{company}")
async def delete_policy(company: str):
    try:
        db_client.delete_policy(company)
        return JSONResponse({"message": f"Policy deleted for {company}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
