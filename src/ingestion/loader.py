from pathlib import Path
from pypdf import PdfReader
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    logger.info(f"PDF loaded: {file_path} | pages : {len(reader.pages)}")
    return text.strip()

def load_txt(file_path: str) -> str:
    """Read text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    logger.info(f"TXT loaded: {file_path} | characters: {len(text)}")
    return text

def load_document(file_path: str) -> str:
    """Load a PDF or TXT document and return its text content."""
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    logger.debug(f"Loading document: {file_path} | extension: {ext}")

    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.txt':
        return load_txt(file_path)
    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")
    
    