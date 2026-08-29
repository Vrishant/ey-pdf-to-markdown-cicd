import re
from typing import List

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from ..config import PipelineConfig
from ..ingestion import PDFIngestor
from ..utils import estimate_max_tokens


def build_quant_config(config: PipelineConfig):
    if config.quantization_bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    if config.quantization_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class TableParser:
    """
    Flow:
      1. YOLO detects a table bbox on the page.
      2. We render a high-res crop of that bbox.
      3. If the crop is short enough, send it to Qwen in one shot.
      4. If the crop is too tall, slice it into horizontal strips,
         stitch the header row onto each strip, send each to Qwen,
         and concatenate all the HTML row outputs.
    """

    FULL_TABLE_PROMPT = (
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

    STRIP_PROMPT = (
        "This image shows a SECTION of a larger table (section {i} of {n}). "
        "The column headers are visible at the top of the image for reference. "
        "Extract ONLY the data rows below the header — do NOT re-extract the "
        "header rows themselves. "
        "Output each data row as <tr><td>...</td>...</tr>. "
        "If a row is partially cut off at the bottom edge, OMIT it. "
        "Do not invent data. Output strictly raw HTML <tr> elements, no commentary."
    )

    HEADER_PROMPT = (
        "This image shows ONLY the header portion of a table. "
        "Extract the header rows as HTML: use <tr><th>...</th>...</tr> for each "
        "header row. If cells span multiple columns or rows, use colspan/rowspan. "
        "Output ONLY <tr> elements, no <table>/<thead> wrappers, no commentary."
    )

    def __init__(self, config: PipelineConfig, ingestor: PDFIngestor, layout_extractor=None):
        self.config = config
        self.ingestor = ingestor
        self.layout_extractor = layout_extractor  # kept for future use
        quant_config = build_quant_config(config)
        device_map = {"": "cuda"} if torch.cuda.is_available() else "auto"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.qwen_dir,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            device_map=device_map,
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(config.qwen_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def parse(self, pdf_path: str, page_num: int, element: dict) -> List[str]:
        """Crop the table, decide whether to split, return HTML blocks."""
        crop = self.ingestor.render_high_res_crop(pdf_path, page_num, element["bbox"])

        # If the crop is short enough, process in one shot
        if crop.height <= self.config.table_split_threshold_px:
            max_tokens = estimate_max_tokens(crop, self.config.max_new_tokens_table)
            raw = self._run_qwen(crop, self.FULL_TABLE_PROMPT, max_tokens)
            return self._split_tables(self._strip_fences(raw))

        # Otherwise, split by height and process each strip
        return [self._parse_large_table(crop)]

    # ------------------------------------------------------------------
    # Large table: slice -> stitch header -> process each strip -> combine
    # ------------------------------------------------------------------
    def _parse_large_table(self, crop: Image.Image) -> str:
        """
        1. Cut header from the top of the crop.
        2. Slice the remainder into horizontal strips.
        3. For each strip, paste the header on top, send to Qwen.
        4. Concatenate all row outputs into one <table>.
        """
        # --- Step 1: Extract header region ---
        header_h = min(int(crop.height * 0.15), 400)
        header_h = max(header_h, 80)  # at least 80px
        header_img = crop.crop((0, 0, crop.width, header_h))

        # --- Step 2: Extract header HTML once ---
        header_tokens = estimate_max_tokens(header_img, self.config.max_new_tokens_table)
        header_raw = self._run_qwen(header_img, self.HEADER_PROMPT, header_tokens)
        header_rows_html = re.findall(
            r"<tr\b.*?</tr>", self._strip_fences(header_raw),
            flags=re.DOTALL | re.IGNORECASE
        )

        # --- Step 3: Slice the body into strips ---
        strip_h = self.config.table_split_threshold_px  # reuse as strip height
        overlap = int(self.config.table_split_overlap_pt * (self.config.extraction_dpi / 72.0))

        strips: List[Image.Image] = []
        top = header_h
        while top < crop.height - 10:
            bottom = min(top + strip_h, crop.height)
            band = crop.crop((0, top, crop.width, bottom))

            # Stitch: header on top + band below
            stitched = Image.new("RGB", (crop.width, header_h + band.height))
            stitched.paste(header_img, (0, 0))
            stitched.paste(band, (0, header_h))
            strips.append(stitched)

            if bottom >= crop.height:
                break
            next_top = bottom - overlap
            if next_top <= top:  # guard infinite loop
                break
            top = next_top

        # --- Step 4: Process each strip ---
        body_rows: List[str] = []
        n = len(strips)
        for i, strip_img in enumerate(strips):
            prompt = self.STRIP_PROMPT.format(i=i + 1, n=n)
            max_tokens = estimate_max_tokens(strip_img, self.config.max_new_tokens_table)
            raw = self._run_qwen(strip_img, prompt, max_tokens)
            rows = re.findall(
                r"<tr\b.*?</tr>", self._strip_fences(raw),
                flags=re.DOTALL | re.IGNORECASE
            )
            body_rows.extend(rows)

        # --- Step 5: Deduplicate consecutive duplicate rows (from overlap) ---
        deduped: List[str] = []
        for row in body_rows:
            norm = re.sub(r"\s+", " ", row).strip()
            if deduped and re.sub(r"\s+", " ", deduped[-1]).strip() == norm:
                continue
            deduped.append(row)

        # --- Step 6: Assemble final HTML ---
        thead = f"<thead>{''.join(header_rows_html)}</thead>" if header_rows_html else ""
        tbody = f"<tbody>{''.join(deduped)}</tbody>"
        return f"<table>{thead}{tbody}</table>"

    # ------------------------------------------------------------------
    # Qwen inference
    # ------------------------------------------------------------------
    def _run_qwen(self, crop: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _split_tables(text: str) -> List[str]:
        tables = re.findall(r"<table.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)
        return tables if tables else ([text.strip()] if text.strip() else [])

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = re.sub(r"^```html\s*|^```\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        return text.strip()