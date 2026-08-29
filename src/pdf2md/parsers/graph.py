from typing import Dict, Tuple

from PIL import Image

from ..config import PipelineConfig
from ..ingestion import PDFIngestor
from ..utils import estimate_max_tokens
from .table import TableParser


class GraphParser:
    COMBINED_PROMPT = (
        "Look at this image and respond in exactly this format:\n"
        "LINE 1: the single word DATA (if it is a chart, graph, plot, or diagram "
        "encoding numeric/categorical data) or DECORATIVE (logo, photo, icon, "
        "screenshot, or non-data diagram).\n"
        "LINE 2 onward: if DATA, extract the underlying values as a valid HTML "
        "<table> (<thead>/<tbody>/<tr>/<th>/<td>), using axis/legend labels as "
        "headers, estimating values as precisely as possible, inventing nothing. "
        "If DECORATIVE, write a 1-3 sentence factual description instead."
    )

    def __init__(self, config: PipelineConfig, ingestor: PDFIngestor, table_parser: TableParser):
        self.config = config
        self.ingestor = ingestor
        self.model = table_parser.model
        self.processor = table_parser.processor

    def parse(self, pdf_path: str, page_num: int, element: Dict) -> Tuple[str, str]:
        crop = self.ingestor.render_high_res_crop(pdf_path, page_num, element["bbox"])
        base_tokens = max(self.config.max_new_tokens_table, self.config.max_new_tokens_figure)
        max_tokens = estimate_max_tokens(crop, base_tokens)
        raw = self._run_qwen(crop, self.COMBINED_PROMPT, max_tokens)
        return self._parse_response(raw)

    def _run_qwen(self, crop: Image.Image, prompt: str, max_new_tokens: int) -> str:
        from qwen_vl_utils import process_vision_info
        import torch

        messages = [{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": prompt}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=self.config.qwen_min_pixels,
            max_pixels=self.config.qwen_max_pixels,
        ).to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample, temperature=self.config.temperature,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _parse_response(raw: str) -> Tuple[str, str]:
        lines = raw.strip().split("\n", 1)
        first = lines[0].strip().upper()
        rest = lines[1].strip() if len(lines) > 1 else ""
        kind = "data" if "DATA" in first else "decorative"
        return kind, (rest if rest else raw)
        