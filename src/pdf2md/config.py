# # src/pdf2md/config.py
# import os
# from dataclasses import dataclass, field
# from typing import Optional


# @dataclass
# class PipelineConfig:
#     # --- Paths ---
#     model_dir: str = "/kaggle/working/models"
#     qwen_dir: str = field(init=False)
#     yolo_dir: str = field(init=False)

#     # --- Rendering (dual-resolution strategy) ---
#     layout_dpi: int = 72
#     extraction_dpi: int = 300

#     # --- Layout detection ---
#     yolo_conf_threshold: float = 0.15
#     min_box_dim_px: int = 20
#     box_merge_y_threshold: int = 20

#     # --- Qwen generation ---
#     max_new_tokens_table: int = 1500
#     max_new_tokens_figure: int = 400
#     do_sample: bool = False
#     temperature: Optional[float] = None

#     # --- Inference optimization ---
#     quantization_bits: Optional[int] = 8   # 8, 4, or None (full fp16)

#     # --- Behavior toggles ---
#     ocr_fallback_enabled: bool = True

#     def __post_init__(self):
#         self.qwen_dir = f"{self.model_dir}/qwen2-vl-2b"
#         self.yolo_dir = f"{self.model_dir}/doclayout-yolo"
#         os.makedirs(self.model_dir, exist_ok=True)

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    model_dir: str = "/kaggle/working/models"
    qwen_dir: str = field(init=False)
    yolo_dir: str = field(init=False)

    layout_dpi: int = 72
    extraction_dpi: int = 300

    yolo_conf_threshold: float = 0.15
    min_box_dim_px: int = 20
    box_merge_y_threshold: int = 20

    max_new_tokens_table: int = 1500
    max_new_tokens_figure: int = 400
    do_sample: bool = False
    temperature: Optional[float] = None

    quantization_bits: Optional[int] = 8

    # Large-table splitting (OOM mitigation & multi-column tracking)
    # Dense financial tables overwhelm Qwen2-VL 2B even at moderate heights.
    # We split any crop taller than 800px into strips with headers stitched on.
    table_split_threshold_px: int = 800           # crops taller than this get split
    qwen_max_pixels: int = 2_000_000
    qwen_min_pixels: int = 3_136
    table_split_band_pt: int = 400
    table_split_overlap_pt: int = 20

    ocr_fallback_enabled: bool = True

    def __post_init__(self):
        self.qwen_dir = f"{self.model_dir}/qwen2-vl-2b"
        self.yolo_dir = f"{self.model_dir}/doclayout-yolo"
        os.makedirs(self.model_dir, exist_ok=True)