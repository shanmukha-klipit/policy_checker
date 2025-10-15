# services/bill_parser.py - Extract structured facts from bills

import google.generativeai as genai
import os
import json
import re
from typing import Dict, Any
from datetime import datetime
import logging
from dotenv import load_dotenv 

load_dotenv()

logger = logging.getLogger(__name__)

class BillParser:
    """
    Parses bill/invoice documents to extract structured facts.
    Uses Gemini LLM to handle various bill formats.
    """
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "your-api-key-here")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Bill Parser initialized")
    
    def parse_bill(self, bill_text: str) -> Dict[str, Any]:
        """
        Extract all relevant facts from bill text.
        Returns structured bill object with metadata.
        """
        
        prompt = f"""You are an expert at extracting information from bills and invoices.

Extract ALL relevant information from this bill/invoice and return it as JSON.

Required fields:
- category: Travel|Accommodation|Food|Communication|Medical|Other
- amount: numeric value (extract the total/final amount)
- currency: INR|USD|EUR|etc (default INR)
- vendor: company/merchant name
- date: transaction date (YYYY-MM-DD format if available)

Optional fields (include if present):
- mode: flight|train|bus|taxi|car|other (for travel)
- origin: starting location (for travel)
- destination: ending location (for travel)
- nights: number of nights (for accommodation)
- hotel_name: name of hotel (for accommodation)
- meal_type: breakfast|lunch|dinner (for food)
- description: brief description of expense
- invoice_number: bill/invoice number
- gst_number: GST/tax ID if present

BILL TEXT:
{bill_text}

Return ONLY valid JSON in this exact format:
{{
  "category": "...",
  "amount": 0.0,
  "currency": "INR",
  "vendor": "...",
  "date": "YYYY-MM-DD",
  "mode": "...",
  "origin": "...",
  "destination": "...",
  "description": "...",
  "invoice_number": "..."
}}

Extract as much as possible. Use "N/A" for unavailable fields."""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            bill_meta = json.loads(result_text)
            
            # Generate unique bill ID
            bill_id = self._generate_bill_id(bill_meta)
            
            # Structure final bill object
            bill_facts = {
                "bill_id": bill_id,
                "bill_meta": bill_meta,
                "raw_text": bill_text,
                "parsed_at": datetime.utcnow().isoformat(),
                "confidence": 0.9  # High confidence for LLM extraction
            }
            
            logger.info(f"Successfully parsed bill: {bill_id}")
            return bill_facts
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {result_text}")
            return self._fallback_extraction(bill_text)
        
        except Exception as e:
            logger.error(f"Error parsing bill: {e}")
            return self._fallback_extraction(bill_text)
    
    def _fallback_extraction(self, bill_text: str) -> Dict[str, Any]:
        """
        Fallback extraction using regex patterns.
        Used when LLM parsing fails.
        """
        
        bill_meta = {
            "category": "Other",
            "amount": 0.0,
            "currency": "INR",
            "vendor": "Unknown",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "description": bill_text[:200]
        }
        
        # Extract amount
        amount_patterns = [
            r'(?:total|amount|rs\.?|inr|₹)\s*:?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:inr|rs)',
            r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, bill_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                bill_meta['amount'] = float(amount_str)
                break
        
        # Extract vendor/company name (look for common patterns)
        vendor_patterns = [
            r'(?:from|vendor|company|issued by)\s*:?\s*([A-Z][A-Za-z\s&.]+)',
            r'^([A-Z][A-Za-z\s&.]{3,30})',  # Capitalized name at start
        ]
        
        for pattern in vendor_patterns:
            match = re.search(pattern, bill_text)
            if match:
                bill_meta['vendor'] = match.group(1).strip()
                break
        
        # Detect category from keywords
        bill_lower = bill_text.lower()
        if any(kw in bill_lower for kw in ['flight', 'airline', 'ticket', 'train', 'bus']):
            bill_meta['category'] = 'Travel'
            
            # Try to extract mode
            if 'flight' in bill_lower or 'airline' in bill_lower:
                bill_meta['mode'] = 'flight'
            elif 'train' in bill_lower:
                bill_meta['mode'] = 'train'
            elif 'bus' in bill_lower:
                bill_meta['mode'] = 'bus'
        
        elif any(kw in bill_lower for kw in ['hotel', 'accommodation', 'stay', 'room']):
            bill_meta['category'] = 'Accommodation'
        
        elif any(kw in bill_lower for kw in ['restaurant', 'food', 'meal', 'cafe']):
            bill_meta['category'] = 'Food'
        
        bill_id = self._generate_bill_id(bill_meta)
        
        logger.warning(f"Used fallback extraction for bill: {bill_id}")
        
        return {
            "bill_id": bill_id,
            "bill_meta": bill_meta,
            "raw_text": bill_text,
            "parsed_at": datetime.utcnow().isoformat(),
            "confidence": 0.5  # Lower confidence for fallback
        }
    
    def _generate_bill_id(self, bill_meta: Dict[str, Any]) -> str:
        """Generate unique bill identifier."""
        import hashlib
        
        # Create hash from key bill attributes
        content = f"{bill_meta.get('vendor', 'unknown')}_{bill_meta.get('amount', 0)}_{bill_meta.get('date', '')}"
        hash_obj = hashlib.md5(content.encode())
        short_hash = hash_obj.hexdigest()[:8]
        
        return f"bill_{short_hash}"
    
    def validate_bill_data(self, bill_facts: Dict[str, Any]) -> bool:
        """
        Validate that extracted bill data meets minimum requirements.
        Returns True if valid, False otherwise.
        """
        
        bill_meta = bill_facts.get('bill_meta', {})
        
        # Must have category
        if not bill_meta.get('category') or bill_meta['category'] == 'Other':
            logger.warning("Bill missing valid category")
            return False
        
        # Must have amount > 0
        if not bill_meta.get('amount') or float(bill_meta['amount']) <= 0:
            logger.warning("Bill missing valid amount")
            return False
        
        # Must have vendor
        if not bill_meta.get('vendor') or bill_meta['vendor'] == 'Unknown':
            logger.warning("Bill missing vendor information")
            return False
        
        return True