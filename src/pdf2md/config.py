# src/pdf2md/config.py
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class PipelineConfig:
    # --- Paths ---
    model_dir: str = "/kaggle/working/models"
    qwen_dir: str = field(init=False)
    yolo_dir: str = field(init=False)

    # --- Rendering (dual-resolution strategy) ---
    layout_dpi: int = 72
    extraction_dpi: int = 300

    # --- Layout detection ---
    yolo_conf_threshold: float = 0.15
    min_box_dim_px: int = 20
    box_merge_y_threshold: int = 20

    # --- Qwen generation ---
    max_new_tokens_table: int = 1500
    max_new_tokens_figure: int = 400
    do_sample: bool = False
    temperature: Optional[float] = None

    # --- Inference optimization ---
    quantization_bits: Optional[int] = 8   # 8, 4, or None (full fp16)

    # --- Behavior toggles ---
    ocr_fallback_enabled: bool = True

    def __post_init__(self):
        self.qwen_dir = f"{self.model_dir}/qwen2-vl-2b"
        self.yolo_dir = f"{self.model_dir}/doclayout-yolo"
        os.makedirs(self.model_dir, exist_ok=True)