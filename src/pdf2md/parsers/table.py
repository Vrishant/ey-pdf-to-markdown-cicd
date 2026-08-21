# import re
# from typing import List

# import torch
# from PIL import Image
# from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info

# from ..config import PipelineConfig
# from ..ingestion import PDFIngestor
# from ..utils import estimate_max_tokens


# def build_quant_config(config: PipelineConfig):
#     if config.quantization_bits == 4:
#         return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
#     if config.quantization_bits == 8:
#         return BitsAndBytesConfig(load_in_8bit=True)
#     return None


# class TableParser:
#     SYSTEM_PROMPT = (
#         "You are a strict data extraction engine. Extract ALL text, numbers, "
#         "and data from ANY AND ALL tables present in this image. Reconstruct "
#         "them as valid HTML <table> elements using <thead>/<tbody>/<tr>/<th>/<td>. "
#         "If a cell visually spans multiple rows or columns (a merged cell), "
#         "reproduce that with the correct rowspan and colspan attributes rather "
#         "than repeating or blanking the value. "
#         "If there are multiple distinct tables in the image, output multiple "
#         "sequential <table> blocks, each fully self-contained — never combine "
#         "rows from different tables into one <table>. "
#         "Do NOT invent placeholder values. Output strictly raw HTML, no commentary."
#     )

#     def __init__(self, config: PipelineConfig, ingestor: PDFIngestor):
#         self.config = config
#         self.ingestor = ingestor
#         quant_config = build_quant_config(config)
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             config.qwen_dir,
#             torch_dtype=None if quant_config else torch.float16,
#             quantization_config=quant_config,
#             device_map="auto",
#         )
#         self.processor = AutoProcessor.from_pretrained(config.qwen_dir)

#     def parse(self, pdf_path: str, page_num: int, element: dict) -> List[str]:
#         crop = self.ingestor.render_high_res_crop(pdf_path, page_num, element["bbox"])
#         max_tokens = estimate_max_tokens(crop, self.config.max_new_tokens_table)
#         raw_output = self._run_qwen(crop, self.SYSTEM_PROMPT, max_tokens)
#         return self._split_tables(self._strip_fences(raw_output))

#     def _run_qwen(self, crop: Image.Image, prompt: str, max_new_tokens: int) -> str:
#         messages = [{"role": "user", "content": [
#             {"type": "image", "image": crop},
#             {"type": "text", "text": prompt},
#         ]}]
#         text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, _ = process_vision_info(messages)
#         inputs = self.processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to(self.model.device)
#         with torch.no_grad():
#             generated_ids = self.model.generate(
#                 **inputs, max_new_tokens=max_new_tokens,
#                 do_sample=self.config.do_sample, temperature=self.config.temperature,
#             )
#         trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
#         return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

#     @staticmethod
#     def _split_tables(text: str) -> List[str]:
#         tables = re.findall(r"<table.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)
#         return tables if tables else ([text.strip()] if text.strip() else [])

#     @staticmethod
#     def _strip_fences(text: str) -> str:
#         text = re.sub(r"^```html\s*|^```\s*", "", text)
#         text = re.sub(r"```\s*$", "", text)
#         return text.strip()

####### SPLIT AND MERGE CODE #######

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

    BAND_PROMPT = (
        "This image is a VERTICAL SECTION of one larger table (section {i} of {n}), "
        "cropped from a taller original — it is not necessarily the whole table. "
        "Extract each fully visible row as a separate HTML row: use <tr><th>...</th>...</tr> "
        "for a column-header row, or <tr><td>...</td>...</tr> for a data row. "
        "Output ONLY the <tr> elements, one after another — NO <table>, <thead>, or "
        "<tbody> wrapper tags. "
        "If a row is only partially visible (visibly cut off at the very top or bottom "
        "edge of this image), OMIT that row entirely — it will be fully visible in a "
        "neighboring section instead. Do not invent data. Output strictly raw HTML "
        "<tr> elements, no commentary."
    )

    def __init__(self, config: PipelineConfig, ingestor: PDFIngestor, layout_extractor=None):
        self.config = config
        self.ingestor = ingestor
        self.layout_extractor = layout_extractor
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

    def parse(self, pdf_path: str, page_num: int, element: dict) -> List[str]:
        crop = self.ingestor.render_high_res_crop(pdf_path, page_num, element["bbox"])
        crop_pixels = crop.width * crop.height
        
        safe_pixels = self.config.qwen_max_pixels
        bbox = element["bbox"]
        table_height_pt = bbox[3] - bbox[1]
        if crop_pixels <= safe_pixels and table_height_pt <= self.config.table_split_band_pt:
            max_tokens = estimate_max_tokens(crop, self.config.max_new_tokens_table)
            raw_output = self._run_qwen(crop, self.SYSTEM_PROMPT, max_tokens)
            return self._split_tables(self._strip_fences(raw_output))
            
        return [self._parse_large_table(pdf_path, page_num, element["bbox"])]

    def _parse_large_table(self, pdf_path: str, page_num: int, bbox: list) -> str:
        # Render the ENTIRE table crop at high resolution
        crop = self.ingestor.render_high_res_crop(pdf_path, page_num, bbox)
        
        # 1. Extract Header Crop (Top 15%, max 450 pixels)
        header_h_px = min(int(crop.height * 0.15), 450)
        header_crop = crop.crop((0, 0, crop.width, header_h_px))
        
        # 2. Determine Band Height (pixels) to stay under qwen_max_pixels
        safe_pixels = self.config.qwen_max_pixels
        band_max_pixels = safe_pixels - (crop.width * header_h_px)
        band_h_px = max(int(band_max_pixels / crop.width), 100) # At least 100px
        
        # Also respect table_split_band_pt constraint
        scale = self.config.extraction_dpi / 72.0
        config_band_h_px = int(self.config.table_split_band_pt * scale)
        band_h_px = min(band_h_px, config_band_h_px)
        
        overlap_px = int(self.config.table_split_overlap_pt * scale)
        
        # 3. Create Stitched Bands
        bands = []
        top = header_h_px
        while top < crop.height - 10:
            bottom = min(top + band_h_px, crop.height)
            band_crop = crop.crop((0, top, crop.width, bottom))
            
            # Stitch Header + Band vertically
            stitched = Image.new('RGB', (crop.width, header_h_px + band_crop.height))
            stitched.paste(header_crop, (0, 0))
            stitched.paste(band_crop, (0, header_h_px))
            bands.append(stitched)
            
            if bottom >= crop.height:
                break
            top = bottom - overlap_px
            if top >= bottom: # safeguard
                break

        # 4. Process each stitched band
        all_rows: List[str] = []
        for i, stitched_crop in enumerate(bands):
            prompt = self.BAND_PROMPT.format(i=i + 1, n=len(bands))
            max_tokens = estimate_max_tokens(stitched_crop, self.config.max_new_tokens_table)
            raw = self._run_qwen(stitched_crop, prompt, max_tokens)
            rows = re.findall(r"<tr\b.*?</tr>", self._strip_fences(raw), flags=re.DOTALL | re.IGNORECASE)
            all_rows.extend(rows)

        return self._merge_rows(all_rows)

    @staticmethod
    def _merge_rows(rows: List[str]) -> str:
        # Overlap between bands can produce duplicate identical <tr> strings
        # for rows that appeared fully in two consecutive bands — drop exact
        # consecutive duplicates rather than a full-document dedupe, since
        # legitimately identical data rows elsewhere in the table should stay.
        deduped: List[str] = []
        for row in rows:
            normalized = re.sub(r"\s+", " ", row).strip()
            if deduped and re.sub(r"\s+", " ", deduped[-1]).strip() == normalized:
                continue
            deduped.append(row)

        header_rows = [r for r in deduped if re.search(r"<th\b", r, flags=re.IGNORECASE)]
        body_rows = [r for r in deduped if r not in header_rows]

        # Globally deduplicate header rows since every stitched band will produce them
        unique_headers = []
        seen = set()
        for r in header_rows:
            normalized = re.sub(r"\s+", " ", r).strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_headers.append(r)

        thead = f"<thead>{''.join(unique_headers)}</thead>" if unique_headers else ""
        tbody = f"<tbody>{''.join(body_rows)}</tbody>"
        return f"<table>{thead}{tbody}</table>"

    def _run_qwen(self, crop: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": prompt},
        ]}]
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
    def _split_tables(text: str) -> List[str]:
        tables = re.findall(r"<table.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)
        return tables if tables else ([text.strip()] if text.strip() else [])

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = re.sub(r"^```html\s*|^```\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        return text.strip()