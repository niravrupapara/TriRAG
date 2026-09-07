import os
from pypdf import PdfReader
from .base import BaseLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseLoader):

    def load(self, file_path):
        logger.info(f"Loading PDF document: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file does not exist: {file_path}")
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        logger.info(f"Successfully extracted {len(text)} characters across {len(reader.pages)} pages")
        return text.strip()