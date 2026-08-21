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

    def parse(self, pdf_path: str, page_num: int, element: dict, allow_sub_detection: bool = True) -> List[str]:
        crop = self.ingestor.render_high_res_crop(pdf_path, page_num, element["bbox"])
        crop_pixels = crop.width * crop.height
        
        if allow_sub_detection and self.layout_extractor:
            sub_elements = self.layout_extractor.segment_page(crop)
            sub_tables = [el for el in sub_elements if el["type"] == "table"]
            
            valid_sub_tables = []
            for t in sub_tables:
                w = t["bbox"][2] - t["bbox"][0]
                h = t["bbox"][3] - t["bbox"][1]
                if (w * h) / crop_pixels < 0.90:  # Ignore bboxes covering almost the entire crop
                    valid_sub_tables.append(t)
                    
            if len(valid_sub_tables) > 1:
                results = []
                scale = self.config.extraction_dpi / 72.0
                basex, basey = element["bbox"][0], element["bbox"][1]
                
                valid_sub_tables = sorted(valid_sub_tables, key=lambda x: x["bbox"][1])
                
                for sub_el in valid_sub_tables:
                    sx1, sy1, sx2, sy2 = sub_el["bbox"]
                    new_bbox = [
                        basex + sx1 / scale,
                        basey + sy1 / scale,
                        basex + sx2 / scale,
                        basey + sy2 / scale
                    ]
                    new_element = {"type": "table", "bbox": new_bbox}
                    results.extend(self.parse(pdf_path, page_num, new_element, allow_sub_detection=False))
                return results

        safe_pixels = self.config.qwen_max_pixels
        bbox = element["bbox"]
        table_height_pt = bbox[3] - bbox[1]
        if crop_pixels <= safe_pixels and table_height_pt <= self.config.table_split_band_pt:
            max_tokens = estimate_max_tokens(crop, self.config.max_new_tokens_table)
            raw_output = self._run_qwen(crop, self.SYSTEM_PROMPT, max_tokens)
            return self._split_tables(self._strip_fences(raw_output))
            
        return [self._parse_large_table(pdf_path, page_num, element["bbox"])]

    def _parse_large_table(self, pdf_path: str, page_num: int, bbox: list) -> str:
        bands = self._compute_bands(bbox)
        all_rows: List[str] = []

        for i, band_bbox in enumerate(bands):
            crop = self.ingestor.render_high_res_crop(pdf_path, page_num, band_bbox)
            prompt = self.BAND_PROMPT.format(i=i + 1, n=len(bands))
            max_tokens = estimate_max_tokens(crop, self.config.max_new_tokens_table)
            raw = self._run_qwen(crop, prompt, max_tokens)
            rows = re.findall(r"<tr\b.*?</tr>", self._strip_fences(raw), flags=re.DOTALL | re.IGNORECASE)
            all_rows.extend(rows)

        return self._merge_rows(all_rows)

    def _compute_bands(self, bbox: list) -> List[list]:
        x1, y1, x2, y2 = bbox
        
        w_pt = x2 - x1
        scale = self.config.extraction_dpi / 72.0
        w_px = w_pt * scale
        
        # Calculate max band height that keeps pixels within safe limits
        max_band_h = self.config.qwen_max_pixels / (w_px * scale)
        
        overlap = self.config.table_split_overlap_pt
        band_h = min(self.config.table_split_band_pt, max_band_h)
        band_h = max(band_h, overlap + 10)  # Ensure progress is made

        bands = []
        top = y1
        while top < y2 - 1:  # 1pt tolerance to avoid float drift re-entering
            bottom = min(top + band_h, y2)
            bands.append([x1, top, x2, bottom])
            if bottom >= y2:
                break
            next_top = bottom - overlap
            if next_top <= top:  # safety: overlap >= band_h would cause infinite loop
                break
            top = next_top
        return bands

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

        thead = f"<thead>{''.join(header_rows)}</thead>" if header_rows else ""
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