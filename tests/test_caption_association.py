from pdf2md.ordering import CaptionAssociator


def test_table_caption_attaches_to_nearest_table():
    elements = [
        {"type": "table", "bbox": [0, 0, 100, 50]},
        {"type": "table_caption", "bbox": [0, 51, 100, 60]},
        {"type": "table", "bbox": [0, 200, 100, 250]},
    ]
    result = CaptionAssociator().associate(elements)
    tables = [e for e in result if e["type"] == "table"]
    assert len(tables[0].get("captions", [])) == 1
    assert len(tables[1].get("captions", [])) == 0


def test_orphaned_caption_falls_through():
    elements = [{"type": "figure_caption", "bbox": [0, 0, 100, 10]}]
    result = CaptionAssociator().associate(elements)
    assert len(result) == 1
    assert result[0]["type"] == "figure_caption"
