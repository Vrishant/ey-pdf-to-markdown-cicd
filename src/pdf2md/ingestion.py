import io
from typing import Optional

import fitz
from PIL import Image

from .config import PipelineConfig


class PDFIngestor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.layout_matrix = fitz.Matrix(config.layout_dpi / 72, config.layout_dpi / 72)
        self.extraction_matrix = fitz.Matrix(config.extraction_dpi / 72, config.extraction_dpi / 72)
        self._doc: Optional[fitz.Document] = None
        self._doc_path: Optional[str] = None

    def _get_doc(self, pdf_path: str) -> fitz.Document:
        if self._doc is None or self._doc_path != pdf_path:
            if self._doc is not None:
                self._doc.close()
            self._doc = fitz.open(pdf_path)
            self._doc_path = pdf_path
        return self._doc

    def close(self):
        if self._doc is not None:
            self._doc.close()
            self._doc = None
            self._doc_path = None

    def page_count(self, pdf_path: str) -> int:
        return len(self._get_doc(pdf_path))

    def render_layout_canvas(self, pdf_path: str, page_num: int) -> Image.Image:
        page = self._get_doc(pdf_path).load_page(page_num)
        pix = page.get_pixmap(matrix=self.layout_matrix)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    def render_high_res_crop(self, pdf_path: str, page_num: int, bbox_layout_px: list) -> Image.Image:
        x1, y1, x2, y2 = bbox_layout_px
        rect = fitz.Rect(x1, y1, x2, y2)
        page = self._get_doc(pdf_path).load_page(page_num)
        pix = page.get_pixmap(matrix=self.extraction_matrix, clip=rect)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    def get_native_text(self, pdf_path: str, page_num: int, bbox_layout_px: list) -> str:
        x1, y1, x2, y2 = bbox_layout_px
        rect = fitz.Rect(x1, y1, x2, y2)
        page = self._get_doc(pdf_path).load_page(page_num)
        return page.get_text("text", clip=rect).strip()

    def get_text_dict(self, pdf_path: str, page_num: int, bbox_layout_px: list) -> dict:
        x1, y1, x2, y2 = bbox_layout_px
        rect = fitz.Rect(x1, y1, x2, y2)
        page = self._get_doc(pdf_path).load_page(page_num)
        return page.get_text("dict", clip=rect)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
