from typing import Dict

import pytesseract

from ..config import PipelineConfig
from ..ingestion import PDFIngestor


class TextHeadingParser:
    def __init__(self, config: PipelineConfig, ingestor: PDFIngestor):
        self.config = config
        self.ingestor = ingestor

    def parse(self, pdf_path: str, page_num: int, element: Dict) -> str:
        bbox = element["bbox"]
        text = self.ingestor.get_native_text(pdf_path, page_num, bbox)

        if not text and self.config.ocr_fallback_enabled:
            text = self._ocr_fallback(pdf_path, page_num, bbox)

        if not text:
            return ""

        text = text.replace("\n", " ").strip()
        if element["type"] == "heading":
            return f'{"#" * element.get("heading_level", 2)} {text}'
        return text

    def _ocr_fallback(self, pdf_path: str, page_num: int, bbox: list) -> str:
        crop = self.ingestor.render_high_res_crop(pdf_path, page_num, bbox)
        return pytesseract.image_to_string(crop).strip()
