import os
from typing import Any, Dict, List, Optional

from .ingestion import PDFIngestor


class MetadataExtractor:
    def __init__(self, ingestor: PDFIngestor):
        self.ingestor = ingestor

    def extract(self, pdf_path: str, first_page_elements: Optional[List[Dict]] = None) -> Dict[str, Any]:
        doc = self.ingestor._get_doc(pdf_path)
        meta = doc.metadata or {}
        page_count = len(doc)

        title = (meta.get("title") or "").strip()
        if not title and first_page_elements:
            title = self._infer_title_from_layout(pdf_path, first_page_elements)

        return {
            "title": title or os.path.splitext(os.path.basename(pdf_path))[0],
            "author": (meta.get("author") or "").strip(),
            "subject": (meta.get("subject") or "").strip(),
            "creation_date": (meta.get("creationDate") or "").strip(),
            "producer": (meta.get("producer") or "").strip(),
            "page_count": page_count,
            "source_file": os.path.basename(pdf_path),
        }

    def _infer_title_from_layout(self, pdf_path: str, elements: List[Dict]) -> str:
        headings = sorted(
            [e for e in elements if e["type"] == "heading"],
            key=lambda e: e["bbox"][1],
        )
        if not headings:
            return ""
        text = self.ingestor.get_native_text(pdf_path, 0, headings[0]["bbox"])
        return text.replace("\n", " ").strip()
