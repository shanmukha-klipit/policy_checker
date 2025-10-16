# services/pdf_extractor.py - PDF text extraction

import pdfplumber
import PyPDF2
from PIL import Image
import pytesseract
import os
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    Extract text from PDF documents using multiple strategies:
    1. Direct text extraction (pdfplumber)
    2. OCR for scanned documents (pytesseract)
    3. Fallback to PyPDF2
    """
    
    def __init__(self):
        logger.info("PDF Extractor initialized")
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF file using best available method.
        """
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # ext = os.path.splitext(file_path)[1].lower()
        file_type = self._detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type}")


        # If it's an image file
        if file_type == "image":
            return self.extract_from_image(file_path)
        
        # If it's an Excel file
        elif file_type == "excel":
            return self._extract_from_excel_pandas(file_path)

        elif file_type == "pdf":
            # Try pdfplumber first (best for most PDFs)
            text = self._extract_with_pdfplumber(file_path)
            
            if text and len(text.strip()) > 50:
                logger.info(f"Extracted {len(text)} chars using pdfplumber")
                return text
            
            # If minimal text, try PyPDF2
            text = self._extract_with_pypdf2(file_path)
            
            if text and len(text.strip()) > 50:
                logger.info(f"Extracted {len(text)} chars using PyPDF2")
                return text
            
            # If still no text, assume scanned PDF - use OCR
            logger.warning("Minimal text extracted, attempting OCR...")
            text = self._extract_with_ocr(file_path)
            
            if text:
                logger.info(f"Extracted {len(text)} chars using OCR")
                return text
            
            raise ValueError("Could not extract text from PDF")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _extract_with_pdfplumber(self, file_path: str) -> Optional[str]:
        """Extract text using pdfplumber."""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return None
    
    def _extract_with_pypdf2(self, file_path: str) -> Optional[str]:
        """Extract text using PyPDF2."""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return None
    
    def _extract_with_ocr(self, file_path: str) -> Optional[str]:
        """
        Extract text using OCR (for scanned PDFs).
        Requires pytesseract and poppler-utils.
        """
        
        try:
            # Convert PDF to images
            from pdf2image import convert_from_path
            
            images = convert_from_path(file_path)
            text = ""
            
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}")
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
            
            return text.strip()
            
        except ImportError:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            return None
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None
    
    def extract_from_image(self, image_path: str) -> str:
        """Extract text from image file (JPG, PNG)."""
        try:
            # Dynamically set tesseract binary path based on environment
           if os.name == "nt":
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            else:
                pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
                
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            logger.info(f"Extracted {len(text)} chars from image")
            return text.strip()

        except Exception as e:
            logger.error(f"Image text extraction failed: {e}")
            raise

    def get_pdf_info(self, file_path: str) -> dict:
        """Get PDF metadata information."""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                
                return {
                    "pages": len(pdf_reader.pages),
                    "title": info.get('/Title', 'Unknown'),
                    "author": info.get('/Author', 'Unknown'),
                    "creator": info.get('/Creator', 'Unknown'),
                }
        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return {}


    def _detect_file_type(self, file_path: str) -> str:
        """
        Detect whether the file is PDF, Excel, or image based on header bytes only.
        Supports PDF, Excel (.xls, .xlsx), JPEG, PNG, BMP, TIFF, WebP, GIF.
        """
        with open(file_path, "rb") as f:
            header = f.read(12)  # read more bytes for formats like WebP and Excel

        # PDF
        if header.startswith(b"%PDF"):
            logger.info("Detected PDF file from header")
            return "pdf"

        # Excel XLSX (ZIP-based)
        elif header.startswith(b"PK\x03\x04"):
            logger.info("Detected Excel XLSX file from header")
            return "excel"

        # Excel XLS (legacy)
        elif header.startswith(b"\xD0\xCF\x11\xE0"):
            logger.info("Detected Excel XLS file from header")
            return "excel"

        # JPEG
        elif header.startswith(b"\xff\xd8\xff"):
            logger.info("Detected JPEG image from header")
            return "image"

        # PNG
        elif header.startswith(b"\x89PNG"):
            logger.info("Detected PNG image from header")
            return "image"

        # BMP
        elif header.startswith(b"BM"):
            logger.info("Detected BMP image from header")
            return "image"

        # TIFF
        elif header.startswith(b"II") or header.startswith(b"MM"):
            logger.info("Detected TIFF image from header")
            return "image"

        # WebP
        elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            logger.info("Detected WebP image from header")
            return "image"

        # GIF
        elif header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            logger.info("Detected GIF image from header")
            return "image"

        else:
            logger.warning("Unknown file type from header")
            return "unknown"

    def _extract_from_excel_pandas(self, file_path: str) -> Optional[str]:
        """Extract text from all sheets in an Excel file using pandas with complete content extraction."""
        try:
            # Use context manager to ensure file handle is closed
            with pd.ExcelFile(file_path) as xls:
                text = ""
                
                for sheet_name in xls.sheet_names:
                    # Add sheet name as header for clarity
                    text += f"\n{'='*50}\nSheet: {sheet_name}\n{'='*50}\n"
                    
                    # Read with header=None to include all rows (including potential headers)
                    df = pd.read_excel(
                        xls,  # Pass xls object, not file_path
                        sheet_name=sheet_name, 
                        dtype=str,
                        header=None,  # Don't treat first row as header - capture everything
                        na_filter=False,  # Prevent automatic NaN conversion
                        keep_default_na=False  # Don't convert "NA" strings to NaN
                    )
                    
                    # Also read WITH headers to capture column names if they exist
                    df_with_headers = pd.read_excel(
                        xls,  # Pass xls object, not file_path
                        sheet_name=sheet_name,
                        dtype=str,
                        na_filter=False,
                        keep_default_na=False
                    )
                    
                    # Extract column headers if they exist and differ from first row
                    if not df_with_headers.columns.equals(pd.RangeIndex(len(df_with_headers.columns))):
                        headers = "\t".join([str(col) for col in df_with_headers.columns])
                        text += headers + "\n"
                    
                    # Replace NaN and None with empty strings
                    df = df.fillna("")
                    df = df.replace({None: ""})
                    
                    # Convert all cells to strings and strip whitespace
                    for col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
                    
                    # Extract all rows, filtering out completely empty rows
                    for idx, row in df.iterrows():
                        row_text = "\t".join(row.tolist())
                        # Only add non-empty rows
                        if row_text.strip():
                            text += row_text + "\n"
                    
                    text += "\n"  # Add spacing between sheets
            
            # File handle is automatically closed here
            return text.strip()
            
        except Exception as e:
            logger.error(f"Pandas Excel extraction failed: {e}")
            return None