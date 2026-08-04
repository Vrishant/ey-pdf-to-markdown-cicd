import glob
import os
from typing import Any, Dict, List

from PIL import Image
from doclayout_yolo import YOLOv10

from .config import PipelineConfig

DOCSTRUCTBENCH_CLASS_MAP = {
    "title": "heading",
    "plain text": "text",
    "abandon": "ignore",
    "figure": "figure",
    "figure_caption": "figure_caption",
    "table": "table",
    "table_caption": "table_caption",
    "table_footnote": "table_footnote",
    "isolate_formula": "formula",
    "formula_caption": "formula_caption",
}


class LayoutExtractor:
    def __init__(self, config: PipelineConfig):
        pt_files = glob.glob(os.path.join(config.yolo_dir, "*.pt"))
        self.model = YOLOv10(pt_files[0])
        self.conf_threshold = config.yolo_conf_threshold

    def segment_page(self, image: Image.Image) -> List[Dict[str, Any]]:
        results = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        elements = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            raw_class = results.names[class_id].lower().strip()
            el_type = DOCSTRUCTBENCH_CLASS_MAP.get(raw_class, "unknown")
            if el_type == "ignore":
                continue
            elements.append({
                "type": el_type,
                "raw_class": raw_class,
                "bbox": [float(c) for c in box.xyxy[0].tolist()],
                "conf": float(box.conf[0].item()),
            })
        return elements

    def segment_pages_batch(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """Batched multi-page inference (speed optimization A2)."""
        results = self.model(images, conf=self.conf_threshold, verbose=False)
        all_elements = []
        for r in results:
            elements = []
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                raw_class = r.names[class_id].lower().strip()
                el_type = DOCSTRUCTBENCH_CLASS_MAP.get(raw_class, "unknown")
                if el_type == "ignore":
                    continue
                elements.append({
                    "type": el_type,
                    "raw_class": raw_class,
                    "bbox": [float(c) for c in box.xyxy[0].tolist()],
                    "conf": float(box.conf[0].item()),
                })
            all_elements.append(elements)
        return all_elements


class NoiseFilter:
    def __init__(self, config: PipelineConfig):
        self.min_dim = config.min_box_dim_px

    def filter(self, elements: List[Dict]) -> List[Dict]:
        return [
            el for el in elements
            if (el["bbox"][2] - el["bbox"][0]) >= self.min_dim
            or (el["bbox"][3] - el["bbox"][1]) >= self.min_dim
        ]


def merge_neighboring_boxes(elements: List[Dict], y_threshold: int) -> List[Dict]:
    MERGEABLE_TYPES = {"text"}
    by_type: Dict[str, List[Dict]] = {}
    passthrough: List[Dict] = []

    for el in elements:
        if el["type"] in MERGEABLE_TYPES:
            by_type.setdefault(el["type"], []).append(el)
        else:
            passthrough.append(el)

    merged: List[Dict] = list(passthrough)
    for el_type, boxes in by_type.items():
        boxes = sorted(boxes, key=lambda e: e["bbox"][1])
        current = dict(boxes[0])
        for nxt in boxes[1:]:
            if nxt["bbox"][1] <= current["bbox"][3] + y_threshold:
                current["bbox"][0] = min(current["bbox"][0], nxt["bbox"][0])
                current["bbox"][1] = min(current["bbox"][1], nxt["bbox"][1])
                current["bbox"][2] = max(current["bbox"][2], nxt["bbox"][2])
                current["bbox"][3] = max(current["bbox"][3], nxt["bbox"][3])
                current["conf"] = max(current["conf"], nxt["conf"])
            else:
                merged.append(current)
                current = dict(nxt)
        merged.append(current)
    return merged
