from typing import Any, Dict, List


class ReadingOrderSorter:
    def sort_elements(self, elements: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
        midpoint = page_width / 2
        full_width, left_col, right_col = [], [], []

        for el in elements:
            x1, _, x2, _ = el["bbox"]
            width = x2 - x1
            if width > page_width * 0.8:
                full_width.append(el)
            elif x1 < midpoint:
                left_col.append(el)
            else:
                right_col.append(el)

        full_width.sort(key=lambda e: e["bbox"][1])
        left_col.sort(key=lambda e: e["bbox"][1])
        right_col.sort(key=lambda e: e["bbox"][1])

        return sorted(full_width + left_col + right_col, key=lambda e: e["bbox"][1])


class CaptionAssociator:
    def associate(self, elements: List[Dict]) -> List[Dict]:
        captions = [e for e in elements if e["type"] in ("table_caption", "figure_caption", "table_footnote")]
        anchors = [e for e in elements if e["type"] in ("table", "figure")]
        remaining = [e for e in elements if e not in captions and e not in anchors]

        for cap in captions:
            target_type = "table" if "table" in cap["type"] else "figure"
            candidates = [a for a in anchors if a["type"] == target_type]
            if not candidates:
                remaining.append(cap)
                continue
            nearest = min(candidates, key=lambda a: abs(a["bbox"][1] - cap["bbox"][3]))
            nearest.setdefault("captions", []).append(cap)

        return anchors + remaining
