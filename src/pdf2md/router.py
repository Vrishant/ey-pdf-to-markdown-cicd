import gc
import logging
from typing import Dict, List

import torch

from .ingestion import PDFIngestor
from .parsers import TextHeadingParser, TableParser, GraphParser

logger = logging.getLogger("pdf2md")

VRAM_CLEAR_INTERVAL = 5


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ContentRouter:
    def __init__(self, text_parser: TextHeadingParser, table_parser: TableParser,
                 graph_parser: GraphParser, ingestor: PDFIngestor):
        self.text_parser = text_parser
        self.table_parser = table_parser
        self.graph_parser = graph_parser
        self.ingestor = ingestor

    def orchestrate(self, pdf_path: str, page_num: int, elements: List[Dict]) -> List[str]:
        blocks = []
        vlm_call_count = 0

        for el in elements:
            try:
                t = el["type"]
                if t in ("text", "heading"):
                    b = self.text_parser.parse(pdf_path, page_num, el)
                    if b:
                        blocks.append(b)

                elif t == "table":
                    table_htmls = self.table_parser.parse(pdf_path, page_num, el)
                    for table_html in table_htmls:
                        blocks.append(self._wrap(pdf_path, page_num, el, table_html))
                    vlm_call_count += 1

                elif t == "figure":
                    kind, content = self.graph_parser.parse(pdf_path, page_num, el)
                    if kind == "data":
                        blocks.append(self._wrap(pdf_path, page_num, el, content))
                    else:
                        blocks.append(f"*Figure: {content}*")
                    vlm_call_count += 1

                elif t in ("table_caption", "figure_caption", "table_footnote"):
                    b = self.text_parser.parse(pdf_path, page_num, el)
                    if b:
                        blocks.append(b)

            except Exception as e:
                logger.error(f"Failed {el.get('type')} at {el.get('bbox')}: {e}")
            finally:
                if vlm_call_count and vlm_call_count % VRAM_CLEAR_INTERVAL == 0:
                    clear_vram()

        clear_vram()
        return blocks

    def _wrap(self, pdf_path: str, page_num: int, el: Dict, content_html: str) -> str:
        parts = [content_html]
        for cap in el.get("captions", []):
            cap_text = self.ingestor.get_native_text(pdf_path, page_num, cap["bbox"]).replace("\n", " ").strip()
            if cap_text:
                parts.append(f"*{cap_text}*")
        return "\n\n".join(parts)
