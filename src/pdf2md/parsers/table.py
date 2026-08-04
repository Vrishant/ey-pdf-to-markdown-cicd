import re
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from ..config import PipelineConfig
from ..ingestion import PDFIngestor


def build_quant_config(config: PipelineConfig):
    if config.quantization_bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    if config.quantization_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class TableParser:
    SYSTEM_PROMPT = (
        "You are a strict data extraction engine. Extract ALL text, numbers, "
        "and data from ANY AND ALL tables present in this image. Reconstruct "
        "them as valid HTML <table> elements using <thead>/<tbody>/<tr>/<th>/<td>. "
        "If a cell visually spans multiple rows or columns (a merged cell), "
        "reproduce that with the correct rowspan and colspan attributes rather "
        "than repeating or blanking the value. "
        "If there are multiple distinct tables in the image, output multiple "
        "sequential <table> blocks, each fully self-contained — never combine "
        "rows from different tables into one <table>. "
        "Do NOT invent placeholder values. Output strictly raw HTML, no commentary."
    )

    def __init__(self, config: PipelineConfig, ingestor: PDFIngestor):
        self.config = config
        self.ingestor = ingestor
        quant_config = build_quant_config(config)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.qwen_dir,
            torch_dtype=None if quant_config else torch.float16,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(config.qwen_dir)

    def parse(self, pdf_path: str, page_num: int, element: dict) -> List[str]:
        crop = self.ingestor.render_high_res_crop(pdf_path, page_num, element["bbox"])
        raw_output = self._run_qwen(crop, self.SYSTEM_PROMPT, self.config.max_new_tokens_table)
        return self._split_tables(self._strip_fences(raw_output))

    def _run_qwen(self, crop: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample, temperature=self.config.temperature,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _split_tables(text: str) -> List[str]:
        tables = re.findall(r"<table.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)
        return tables if tables else ([text.strip()] if text.strip() else [])

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = re.sub(r"^```html\s*|^```\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        return text.strip()
