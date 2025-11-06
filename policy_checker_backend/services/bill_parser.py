# services/optimized_bill_parser.py - Enhanced & Optimized Bill Parser (URL compatible)

import google.generativeai as genai
import os
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BillParser:
    """
    🚀 ENHANCED Bill parser with:
    - URL-ready bill text processing
    - Smart chunking for large PDFs
    - Auto-fallback to regex if model fails
    - Same output schema for seamless DB integration
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment")

        genai.configure(api_key=api_key)

        # Prefer Pro for accuracy; fallback to Flash if unavailable
        try:
            self.model = genai.GenerativeModel("gemini-2.5-pro")
            logger.info("✅ Using Gemini 2.5 Pro for bill parsing")
        except Exception:
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            logger.warning("⚡ Using fallback model: gemini-2.5-flash")

    # ------------------------- Core Extraction -------------------------

    def parse_bill(self, bill_text: str) -> Dict[str, Any]:
        """
        Extract structured bill information using Gemini LLM.
        Handles long text, malformed PDFs, and ensures JSON-safe output.
        """

        # Clean and trim text for performance
        clean_text = self._normalize_text(bill_text)
        if len(clean_text) > 10000:
            logger.info("✂️ Compressing long bill text for faster processing")
            clean_text = self._summarize_long_text(clean_text)

        prompt = f"""You are a bill information extraction assistant.
Extract the following fields as JSON only (no markdown or explanations).

BILL TEXT:
{clean_text[:8000]}

Return JSON with:
{{
  "category": "Travel|Accommodation|Food|Communication|Medical|Other",
  "amount": number,
  "currency": "INR|USD|EUR",
  "vendor": "company or supplier name",
  "date": "YYYY-MM-DD",
  "mode": "flight|train|bus|taxi|car|other (for travel)",
  "origin": "location (for travel)",
  "destination": "location (for travel)",
  "description": "brief bill description",
  "invoice_number": "bill/invoice number"
}}

If data missing, use "N/A" or 0. Return only valid JSON.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    top_p=0.8,
                    top_k=40
                )
            )

            result_text = (response.text or "").strip()
            result_text = self._extract_json_from_text(result_text)
            bill_meta = json.loads(result_text)

            bill_id = self._generate_bill_id(bill_meta)

            bill_facts = {
                "bill_id": bill_id,
                "bill_meta": bill_meta,
                "raw_text": clean_text,
                "parsed_at": datetime.utcnow().isoformat(),
                "confidence": 0.9
            }

            logger.info(f"✅ Parsed bill {bill_id} ({bill_meta.get('category')})")
            return bill_facts

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            return self._fallback_extraction(clean_text)

        except Exception as e:
            logger.error(f"❌ LLM extraction error: {e}")
            return self._fallback_extraction(clean_text)

    # ------------------------- Fallback Extraction -------------------------

    def _fallback_extraction(self, bill_text: str) -> Dict[str, Any]:
        """Regex-based fast fallback when LLM fails"""

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
            if match := re.search(pattern, bill_text, re.IGNORECASE):
                bill_meta["amount"] = float(match.group(1).replace(",", ""))
                break

        # Extract vendor
        vendor_patterns = [
            r'(?:from|vendor|company|issued by)\s*:?\s*([A-Z][A-Za-z\s&.]+)',
            r'^([A-Z][A-Za-z\s&.]{3,30})',
        ]
        for pattern in vendor_patterns:
            if match := re.search(pattern, bill_text):
                bill_meta["vendor"] = match.group(1).strip()
                break

        # Category detection
        bill_lower = bill_text.lower()
        categories = {
            "Travel": ["flight", "airline", "ticket", "train", "bus"],
            "Accommodation": ["hotel", "accommodation", "stay", "room"],
            "Food": ["restaurant", "meal", "food", "cafe"],
            "Communication": ["mobile", "internet", "phone", "recharge"],
            "Medical": ["pharmacy", "hospital", "clinic", "medicine"]
        }

        for cat, keywords in categories.items():
            if any(k in bill_lower for k in keywords):
                bill_meta["category"] = cat
                if cat == "Travel":
                    if "flight" in bill_lower:
                        bill_meta["mode"] = "flight"
                    elif "train" in bill_lower:
                        bill_meta["mode"] = "train"
                    elif "bus" in bill_lower:
                        bill_meta["mode"] = "bus"
                break

        bill_id = self._generate_bill_id(bill_meta)

        logger.warning(f"⚡ Using fallback parser for bill: {bill_id}")
        return {
            "bill_id": bill_id,
            "bill_meta": bill_meta,
            "raw_text": bill_text,
            "parsed_at": datetime.utcnow().isoformat(),
            "confidence": 0.55
        }

    # ------------------------- Utilities -------------------------

    def _extract_json_from_text(self, text: str) -> str:
        """Clean text and isolate JSON content."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        # Remove trailing commas and non-breaking spaces
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    def _summarize_long_text(self, text: str, max_len: int = 8000) -> str:
        """Trim or summarize long text for model efficiency."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...[truncated for processing]..."

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and remove non-ASCII artifacts."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        return text.strip()

    def _generate_bill_id(self, bill_meta: Dict[str, Any]) -> str:
        """Generate deterministic short hash for bill tracking"""
        import hashlib
        content = f"{bill_meta.get('vendor','')}_{bill_meta.get('amount',0)}_{bill_meta.get('date','')}"
        return "bill_" + hashlib.md5(content.encode()).hexdigest()[:8]

    def validate_bill_data(self, bill_facts: Dict[str, Any]) -> bool:
        """Ensure parsed bill fields are valid before compliance check."""
        bill_meta = bill_facts.get("bill_meta", {})
        if not bill_meta:
            return False
        return all([
            bill_meta.get("amount", 0) > 0,
            bill_meta.get("vendor", "Unknown") != "Unknown",
            bill_meta.get("category", "Other") != "Other"
        ])
