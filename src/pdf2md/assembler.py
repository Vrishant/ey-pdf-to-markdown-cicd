from typing import Any, Dict, List

from .ingestion import PDFIngestor


class MarkdownDocumentAssembler:
    def __init__(self, ingestor: PDFIngestor):
        self.ingestor = ingestor

    def assemble(self, metadata: Dict[str, Any], pages_md: List[List[str]]) -> str:
        frontmatter = self._build_frontmatter(metadata)
        sections = []
        for i, blocks in enumerate(pages_md):
            if not blocks:
                continue
            sections.append(f"<!-- page: {i} -->\n\n" + "\n\n".join(blocks))
        return frontmatter + "\n\n" + "\n\n".join(sections)

    def _build_frontmatter(self, metadata: Dict[str, Any]) -> str:
        lines = ["---"]
        for key in ("title", "author", "source_file", "page_count", "creation_date"):
            val = metadata.get(key, "")
            if val:
                lines.append(f"{key}: {val}")
        lines.append("---")
        return "\n".join(lines)

    def rerank_headings_for_page(self, pdf_path: str, page_num: int, elements: List[Dict]) -> Dict[int, int]:
        heading_els = [e for e in elements if e["type"] == "heading"]
        sizes = []
        for idx, el in enumerate(heading_els):
            text_dict = self.ingestor.get_text_dict(pdf_path, page_num, el["bbox"])
            max_size = max(
                (span.get("size", 0.0) for block in text_dict.get("blocks", [])
                 for line in block.get("lines", []) for span in line.get("spans", [])),
                default=0.0,
            )
            sizes.append((idx, max_size))
        distinct = sorted({s for _, s in sizes}, reverse=True)
        level_map = {size: min(i + 1, 6) for i, size in enumerate(distinct)}
        return {idx: level_map.get(sz, 2) for idx, sz in sizes}
