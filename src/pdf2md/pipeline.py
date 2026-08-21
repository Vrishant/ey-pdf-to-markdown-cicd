import logging
import time
from typing import List, Optional

from .assembler import MarkdownDocumentAssembler
from .config import PipelineConfig
from .ingestion import PDFIngestor
from .layout import LayoutExtractor, NoiseFilter, merge_neighboring_boxes
from .metadata import MetadataExtractor
from .observability import PageMetricsRecorder
from .ordering import CaptionAssociator, ReadingOrderSorter
from .parsers import GraphParser, TableParser, TextHeadingParser
from .router import ContentRouter

logger = logging.getLogger("pdf2md")


class PDF2MarkdownPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ingestor = PDFIngestor(config)
        self.layout_extractor = LayoutExtractor(config)
        self.noise_filter = NoiseFilter(config)
        self.sorter = ReadingOrderSorter()
        self.caption_associator = CaptionAssociator()
        self.metadata_extractor = MetadataExtractor(self.ingestor)

        table_parser = TableParser(config, self.ingestor, self.layout_extractor)
        text_parser = TextHeadingParser(config, self.ingestor)
        graph_parser = GraphParser(config, self.ingestor, table_parser)

        self.router = ContentRouter(text_parser, table_parser, graph_parser, self.ingestor)
        self.assembler = MarkdownDocumentAssembler(self.ingestor)
        self.metrics = PageMetricsRecorder()

    def process_page(self, pdf_path: str, page_num: int) -> List[str]:
        canvas = self.ingestor.render_layout_canvas(pdf_path, page_num)
        raw_elements = self.layout_extractor.segment_page(canvas)
        clean_elements = self.noise_filter.filter(raw_elements)
        merged_elements = merge_neighboring_boxes(clean_elements, self.config.box_merge_y_threshold)
        associated = self.caption_associator.associate(merged_elements)
        sorted_elements = self.sorter.sort_elements(associated, canvas.width)

        level_map = self.assembler.rerank_headings_for_page(pdf_path, page_num, sorted_elements)
        heading_idx = 0
        for el in sorted_elements:
            if el["type"] == "heading":
                el["heading_level"] = level_map.get(heading_idx, 2)
                heading_idx += 1

        return self.router.orchestrate(pdf_path, page_num, sorted_elements)

    def process_document(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        doc_start = time.perf_counter()
        total_pages = self.ingestor.page_count(pdf_path)
        n_pages = min(max_pages, total_pages) if max_pages else total_pages

        first_page_canvas_elements = None
        pages_md = []

        for page_num in range(n_pages):
            page_start = time.perf_counter()
            logger.info(f"Processing page {page_num + 1}/{n_pages}: {pdf_path}")
            error = False
            try:
                blocks = self.process_page(pdf_path, page_num)
            except Exception as e:
                logger.error(f"Page {page_num} failed: {e}")
                blocks = []
                error = True
            pages_md.append(blocks)
            self.metrics.record_page(
                page_num=page_num,
                duration_s=time.perf_counter() - page_start,
                element_count=len(blocks),
                error=error,
            )

            if page_num == 0:
                canvas = self.ingestor.render_layout_canvas(pdf_path, 0)
                first_page_canvas_elements = self.noise_filter.filter(
                    self.layout_extractor.segment_page(canvas)
                )

        metadata = self.metadata_extractor.extract(pdf_path, first_page_elements=first_page_canvas_elements)
        result = self.assembler.assemble(metadata, pages_md)
        self.ingestor.close()
        self.metrics.record_document(pdf_path, time.perf_counter() - doc_start, n_pages)
        return result
