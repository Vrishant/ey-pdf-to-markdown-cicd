from pdf2md.ordering import ReadingOrderSorter


def test_full_width_and_columns_sort_by_y():
    elements = [
        {"type": "text", "bbox": [0, 100, 40, 120]},   # left col
        {"type": "text", "bbox": [60, 100, 100, 120]},  # right col
        {"type": "heading", "bbox": [0, 0, 100, 20]},   # full width, top
    ]
    result = ReadingOrderSorter().sort_elements(elements, page_width=100)
    assert result[0]["bbox"][1] == 0
