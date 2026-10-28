from io import BytesIO
from pypdf import PdfReader


def load_document(file_name: str, file_bytes: bytes) -> str:
    """Extract readable text from PDF or TXT documents."""
    extension = file_name.lower().rsplit(".", 1)[-1]

    if extension == "txt":
        text = file_bytes.decode("utf-8-sig", errors="replace")

    elif extension == "pdf":
        reader = PdfReader(BytesIO(file_bytes))

        if reader.is_encrypted:
            raise ValueError("This PDF is password-protected.")

        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages)

    else:
        raise ValueError("Please upload a PDF or TXT file.")

    text = text.strip()

    if not text:
        raise ValueError(
            "No readable text found. This may be a scanned PDF "
            "that requires OCR."
        )

    return text
