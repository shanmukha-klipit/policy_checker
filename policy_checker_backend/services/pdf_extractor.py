# services/pdf_extractor.py - Enhanced PDF Text Extraction (URL & Cloud Compatible)

import os
import io
import re
import logging
import requests
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import pdfplumber
import PyPDF2
from PIL import Image
import pytesseract
from typing import Optional, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class EnhancedPDFExtractor:
    """
    ✅ Enhanced PDF Extractor with:
    - Google Drive URL support
    - AWS S3 object support
    - Direct HTTP/HTTPS URL support
    - Full fallback chain (pdfplumber → PyPDF2 → OCR)
    """

    def __init__(self):
        # Initialize optional clients
        self.s3_client = self._init_s3_client()
        self.drive_service = True  # placeholder for Drive capability
        logger.info("🚀 Enhanced PDF Extractor initialized (URL & Cloud compatible)")

    # -------------------------------------------------------------------------
    # MAIN TEXT EXTRACTION ENTRY
    # -------------------------------------------------------------------------
    def extract_text(self, file_path: str) -> str:
        """Extract text from a local PDF or Excel/image file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type}")

        if file_type == "image":
            return self.extract_from_image(file_path)
        elif file_type == "excel":
            return self._extract_from_excel_pandas(file_path)
        elif file_type == "pdf":
            text = self._extract_with_pdfplumber(file_path)
            if text and len(text.strip()) > 50:
                return text
            text = self._extract_with_pypdf2(file_path)
            if text and len(text.strip()) > 50:
                return text
            logger.warning("⚠️ Minimal text detected; switching to OCR...")
            return self._extract_with_ocr(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    # -------------------------------------------------------------------------
    # URL SUPPORT
    # -------------------------------------------------------------------------
    def validate_url(self, url: str) -> Dict[str, any]:
        """Identify URL type (Google Drive, S3, or direct) and check accessibility."""
        try:
            if "drive.google.com" in url:
                match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
                if not match:
                    return {"valid": False, "source": "google_drive", "accessible": False, "error": "Invalid Drive URL"}
                file_id = match.group(1)
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                resp = requests.head(download_url, allow_redirects=True)
                return {
                    "valid": True,
                    "source": "google_drive",
                    "accessible": resp.status_code == 200,
                    "error": None if resp.status_code == 200 else f"Drive file not accessible ({resp.status_code})"
                }

            elif url.startswith("s3://"):
                # s3://bucket/key
                parts = url.replace("s3://", "").split("/", 1)
                if len(parts) < 2:
                    return {"valid": False, "source": "aws_s3", "accessible": False, "error": "Invalid S3 URL format"}
                bucket, key = parts
                try:
                    self.s3_client.head_object(Bucket=bucket, Key=key)
                    return {"valid": True, "source": "aws_s3", "accessible": True}
                except NoCredentialsError:
                    return {"valid": True, "source": "aws_s3", "accessible": False, "error": "Missing AWS credentials"}
                except ClientError as e:
                    return {"valid": True, "source": "aws_s3", "accessible": False, "error": str(e)}

            elif url.startswith("https://") or url.startswith("http://"):
                resp = requests.head(url, allow_redirects=True, timeout=10)
                return {
                    "valid": True,
                    "source": "direct_url",
                    "accessible": resp.status_code == 200,
                    "error": None if resp.status_code == 200 else f"URL returned {resp.status_code}"
                }

            return {"valid": False, "source": "unknown", "accessible": False, "error": "Unsupported URL format"}

        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            return {"valid": False, "source": "unknown", "accessible": False, "error": str(e)}

    def extract_from_url(self, url: str) -> str:
        """
        Extract text directly from a remote URL.
        Handles:
        - Google Drive: Converts to download link
        - AWS S3: Reads via boto3 or presigned URL
        - HTTPS/HTTP: Direct download
        """
        logger.info(f"📥 Extracting from URL: {url}")

        if "drive.google.com" in url:
            match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
            if not match:
                raise ValueError("Invalid Google Drive URL")
            file_id = match.group(1)
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url, stream=True)
            if response.status_code != 200:
                raise ValueError(f"Failed to download from Google Drive ({response.status_code})")
            pdf_bytes = io.BytesIO(response.content)

        elif url.startswith("s3://"):
            parts = url.replace("s3://", "").split("/", 1)
            if len(parts) < 2:
                raise ValueError("Invalid S3 URL format")
            bucket, key = parts
            try:
                obj = self.s3_client.get_object(Bucket=bucket, Key=key)
                pdf_bytes = io.BytesIO(obj["Body"].read())
            except Exception as e:
                raise ValueError(f"Failed to read S3 object: {e}")

        elif url.startswith("https://") or url.startswith("http://"):
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch file (HTTP {response.status_code})")
            pdf_bytes = io.BytesIO(response.content)

        else:
            raise ValueError("Unsupported URL format")

        # Try extracting text from in-memory bytes
        try:
            text = self._extract_text_from_bytes(pdf_bytes)
            if text and len(text.strip()) > 50:
                logger.info(f"✅ Extracted {len(text)} chars from remote file")
                return text
            logger.warning("⚠️ Minimal text detected; attempting OCR from bytes")
            return self._extract_ocr_from_bytes(pdf_bytes)
        except Exception as e:
            raise ValueError(f"Failed to extract text from URL: {e}")

    # -------------------------------------------------------------------------
    # HELPERS: In-memory & OCR
    # -------------------------------------------------------------------------
    def _extract_text_from_bytes(self, byte_stream: io.BytesIO) -> str:
        """Extract text from PDF bytes using pdfplumber or PyPDF2."""
        text = ""
        try:
            byte_stream.seek(0)
            with pdfplumber.open(byte_stream) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber (bytes) failed: {e}")
        if not text.strip():
            try:
                byte_stream.seek(0)
                reader = PyPDF2.PdfReader(byte_stream)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                logger.warning(f"PyPDF2 (bytes) failed: {e}")
        return text.strip()

    def _extract_ocr_from_bytes(self, byte_stream: io.BytesIO) -> str:
        """Perform OCR on in-memory PDF bytes."""
        from pdf2image import convert_from_bytes
        text = ""
        try:
            images = convert_from_bytes(byte_stream.read())
            for i, img in enumerate(images):
                logger.info(f"OCR page {i+1}/{len(images)}")
                page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"OCR from bytes failed: {e}")
            return ""

    # -------------------------------------------------------------------------
    # LOCAL EXTRACTION (Unchanged from your version)
    # -------------------------------------------------------------------------
    def _extract_with_pdfplumber(self, file_path: str) -> Optional[str]:
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "".join([p.extract_text() or "" for p in pdf.pages])
            return text.strip()
        except Exception as e:
            logger.error(f"pdfplumber failed: {e}")
            return None

    def _extract_with_pypdf2(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join([p.extract_text() or "" for p in reader.pages])
            return text.strip()
        except Exception as e:
            logger.error(f"PyPDF2 failed: {e}")
            return None

    def _extract_with_ocr(self, file_path: str) -> Optional[str]:
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            text = ""
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None

    def extract_from_image(self, image_path: str) -> str:
        try:
            if os.name == "nt":
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            else:
                pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # S3 & TYPE DETECTION
    # -------------------------------------------------------------------------
    def _init_s3_client(self):
        """Initialize boto3 client if credentials exist."""
        try:
            if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
                return boto3.client("s3")
            return None
        except Exception as e:
            logger.warning(f"S3 client not initialized: {e}")
            return None

    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type (same logic as before)."""
        with open(file_path, "rb") as f:
            header = f.read(12)
        if header.startswith(b"%PDF"):
            return "pdf"
        elif header.startswith(b"PK\x03\x04") or header.startswith(b"\xD0\xCF\x11\xE0"):
            return "excel"
        elif header.startswith(b"\xff\xd8\xff") or header.startswith(b"\x89PNG"):
            return "image"
        return "unknown"

    def _extract_from_excel_pandas(self, file_path: str) -> Optional[str]:
        """Extract text from Excel sheets."""
        try:
            with pd.ExcelFile(file_path) as xls:
                text = ""
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet, dtype=str, header=None, na_filter=False)
                    df = df.fillna("").replace({None: ""})
                    for _, row in df.iterrows():
                        row_text = "\t".join(row.tolist()).strip()
                        if row_text:
                            text += row_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            return None
