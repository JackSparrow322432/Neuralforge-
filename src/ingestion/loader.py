"""Document loader — PDF, web pages, markdown files."""

import hashlib
import io
from typing import Optional

import httpx
from pypdf import PdfReader


class DocumentLoader:
    """Load documents from various sources into raw text."""

    @staticmethod
    async def load_pdf(file_bytes: bytes, filename: str) -> dict:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text.strip()})

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        return {
            "text": full_text,
            "pages": pages,
            "filename": filename,
            "source_type": "pdf",
            "content_hash": content_hash,
            "metadata": {"total_pages": len(reader.pages)},
        }

    @staticmethod
    async def load_url(url: str) -> dict:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()

        # Basic HTML-to-text extraction
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else url
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        return {
            "text": text,
            "filename": title,
            "source_type": "web",
            "content_hash": content_hash,
            "metadata": {"url": url},
        }

    @staticmethod
    async def load_markdown(file_bytes: bytes, filename: str) -> dict:
        text = file_bytes.decode("utf-8", errors="replace")
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        return {
            "text": text,
            "filename": filename,
            "source_type": "markdown",
            "content_hash": content_hash,
            "metadata": {},
        }
