"""
pdf_utils.py
------------
Step 1 and 2 of the RAG pipeline:

    PDF  ->  Text Extraction (per page)  ->  Chunking (with overlap)

Every piece of text is wrapped in a LangChain `Document`, which is simply
    Document(page_content="...text...", metadata={"source": "file.pdf", "page": 3})
Keeping the filename and page number in the metadata is what later allows us
to show source citations next to each answer.
"""

import io
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import SETTINGS


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be opened or read at all."""


@dataclass
class PDFReport:
    """Summary of what happened while reading one PDF (shown in the UI)."""

    filename: str
    total_pages: int = 0
    text_pages: int = 0  # pages that actually contained extractable text
    characters: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.warnings:
            return "warning"
        return "ok"


def clean_text(text: str) -> str:
    """Normalise whitespace produced by PDF extraction."""
    text = text.replace("\x00", "")
    # Re-join words hyphenated across line breaks: "retrie-\nval" -> "retrieval"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of spaces/tabs, and 3+ newlines into a paragraph break.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(file_bytes: bytes, filename: str) -> tuple[list[Document], PDFReport]:
    """
    Read a PDF and return one Document per page that contains text,
    together with a PDFReport describing the result.
    """
    report = PDFReport(filename=filename)

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        # Some PDFs are "encrypted" with an empty password; try to open them.
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFProcessingError("the file is password-protected")
        report.total_pages = len(reader.pages)
    except PDFProcessingError as exc:
        report.error = f"Could not open PDF: {exc}."
        return [], report
    except (PdfReadError, ValueError, OSError) as exc:
        report.error = f"Could not open PDF (file may be corrupted): {exc}"
        return [], report

    if report.total_pages == 0:
        report.error = "The PDF has no pages."
        return [], report

    pages: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:  # a single broken page should not stop the whole file
            text = ""

        if len(text) < SETTINGS.min_chars_per_page:
            continue  # empty or image-only page

        pages.append(
            Document(page_content=text, metadata={"source": filename, "page": page_number})
        )
        report.characters += len(text)

    report.text_pages = len(pages)

    # ---- Quality checks ------------------------------------------------
    if report.text_pages == 0:
        report.error = (
            "No extractable text found. The PDF is probably scanned (images only) "
            "and would need OCR first."
        )
        return [], report

    if report.characters < SETTINGS.min_chars_per_document:
        report.warnings.append(
            f"Very little text was extracted ({report.characters} characters). "
            "Answers from this file may be poor."
        )

    skipped = report.total_pages - report.text_pages
    if skipped > 0:
        report.warnings.append(f"{skipped} page(s) had no extractable text and were skipped.")

    return pages, report


def chunk_documents(
    pages: list[Document],
    chunk_size: int = SETTINGS.default_chunk_size,
    chunk_overlap: int = SETTINGS.default_chunk_overlap,
) -> list[Document]:
    """
    Split page-level Documents into smaller overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on the "most natural"
    boundary first (paragraph -> line -> sentence -> word), so chunks rarely
    cut a sentence in half. The overlap repeats the end of one chunk at the
    start of the next one, so information on a chunk boundary is not lost.

    `split_documents` copies each page's metadata (source + page) to every
    chunk produced from that page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",  # keep "." at the end of a sentence, not at the start of the next chunk
    )
    chunks = splitter.split_documents(pages)

    # Give every chunk a readable id, useful when inspecting retrieved context.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
