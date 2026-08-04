from pdf2md.layout import DOCSTRUCTBENCH_CLASS_MAP, NoiseFilter, merge_neighboring_boxes
from pdf2md.config import PipelineConfig


def test_table_caption_is_not_table():
    assert DOCSTRUCTBENCH_CLASS_MAP["table_caption"] != "table"
    assert DOCSTRUCTBENCH_CLASS_MAP["table_caption"] == "table_caption"


def test_figure_caption_is_not_figure():
    assert DOCSTRUCTBENCH_CLASS_MAP["figure_caption"] != "figure"


def test_abandon_maps_to_ignore():
    assert DOCSTRUCTBENCH_CLASS_MAP["abandon"] == "ignore"


def test_noise_filter_drops_small_boxes():
    cfg = PipelineConfig()
    cfg.min_box_dim_px = 20
    nf = NoiseFilter(cfg)
    elements = [
        {"type": "text", "bbox": [0, 0, 5, 5]},
        {"type": "text", "bbox": [0, 0, 50, 50]},
    ]
    result = nf.filter(elements)
    assert len(result) == 1
    assert result[0]["bbox"] == [0, 0, 50, 50]


def test_merge_never_combines_two_tables():
    elements = [
        {"type": "table", "bbox": [0, 0, 100, 50], "conf": 0.9},
        {"type": "table", "bbox": [0, 55, 100, 100], "conf": 0.9},
    ]
    result = merge_neighboring_boxes(elements, y_threshold=20)
    assert len(result) == 2


def test_merge_combines_fragmented_text():
    elements = [
        {"type": "text", "bbox": [0, 0, 100, 20], "conf": 0.9},
        {"type": "text", "bbox": [0, 25, 100, 45], "conf": 0.9},
    ]
    result = merge_neighboring_boxes(elements, y_threshold=20)
    assert len(result) == 1
