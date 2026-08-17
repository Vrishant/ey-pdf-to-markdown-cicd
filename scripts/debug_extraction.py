# Location: scripts/debug_extraction.py
"""
Diagnoses "zero content extracted" failures by instrumenting the real
pipeline path stage by stage, instead of guessing.

For each page, this:
  1. Renders the layout canvas and runs LayoutExtractor directly, printing
     every detected box (type, raw_class, confidence, bbox) BEFORE any
     filtering — so we can see if the detector found tables at all.
  2. Draws those boxes on the canvas and saves it as a PNG, so detection
     can be checked visually against the source page.
  3. Re-runs NoiseFilter and merge_neighboring_boxes, printing what survives
     each step — so we can see if a table box is being dropped/altered.
  4. For every surviving 'table' element, saves the exact high-res crop
     sent to Qwen, then calls the model directly and prints/saves:
       - the completely raw model output (before strip_fences)
       - the fence-stripped output
       - the final _split_tables() result
     This isolates "Qwen returned nothing/garbage" from
     "Qwen returned a valid table but our regex failed to find it".

Usage:
    python scripts/debug_extraction.py input.pdf --model-dir /kaggle/working/models
"""
import argparse
import os

from PIL import Image, ImageDraw

from pdf2md.config import PipelineConfig
from pdf2md.ingestion import PDFIngestor
from pdf2md.layout import LayoutExtractor, NoiseFilter, merge_neighboring_boxes
from pdf2md.model_fetch import fetch_models
from pdf2md.parsers.table import TableParser

COLOR_MAP = {
    "text": (150, 100, 255),
    "heading": (255, 100, 150),
    "table": (255, 200, 0),
    "figure": (0, 200, 100),
    "table_caption": (255, 140, 0),
    "figure_caption": (0, 150, 255),
    "table_footnote": (150, 150, 150),
    "unknown": (255, 0, 0),
}


def draw_boxes(canvas: Image.Image, elements: list, out_path: str):
    img = canvas.copy()
    draw = ImageDraw.Draw(img)
    for el in elements:
        x1, y1, x2, y2 = el["bbox"]
        color = COLOR_MAP.get(el["type"], (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), f'{el["type"]} ({el["conf"]:.2f})', fill=color)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--model-dir", default="/kaggle/working/models")
    parser.add_argument("--out-dir", default="debug_output")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = PipelineConfig(model_dir=args.model_dir)
    print(f"[setup] Ensuring model weights present in {args.model_dir} ...")
    fetch_models(args.model_dir)

    ingestor = PDFIngestor(config)
    layout_extractor = LayoutExtractor(config)
    noise_filter = NoiseFilter(config)

    n_pages = ingestor.page_count(args.pdf_path)
    if args.max_pages:
        n_pages = min(n_pages, args.max_pages)

    print(f"[setup] Loading TableParser (Qwen2-VL) — this takes a minute ...")
    table_parser = TableParser(config, ingestor)

    for page_num in range(n_pages):
        print(f"\n{'=' * 70}\nPAGE {page_num}\n{'=' * 70}")

        canvas = ingestor.render_layout_canvas(args.pdf_path, page_num)
        raw_elements = layout_extractor.segment_page(canvas)

        print(f"[detect] {len(raw_elements)} raw boxes before any filtering:")
        for el in raw_elements:
            print(f"    type={el['type']:<16} raw_class={el['raw_class']:<16} "
                  f"conf={el['conf']:.3f} bbox={[round(c) for c in el['bbox']]}")

        overlay_path = os.path.join(args.out_dir, f"page{page_num}_raw_detections.png")
        draw_boxes(canvas, raw_elements, overlay_path)
        print(f"[detect] Saved overlay: {overlay_path}")

        clean_elements = noise_filter.filter(raw_elements)
        print(f"[filter] {len(clean_elements)} boxes survive NoiseFilter "
              f"(dropped {len(raw_elements) - len(clean_elements)})")

        merged_elements = merge_neighboring_boxes(clean_elements, config.box_merge_y_threshold)
        print(f"[merge]  {len(merged_elements)} boxes after merge_neighboring_boxes")

        tables = [e for e in merged_elements if e["type"] == "table"]
        if not tables:
            print(f"[RESULT] No 'table' elements survived to this point on page {page_num}. "
                  f"This means the FAILURE IS IN DETECTION, not extraction — "
                  f"check the overlay PNG and/or lower yolo_conf_threshold.")
            continue

        for i, table_el in enumerate(tables):
            print(f"\n  --- table {i} on page {page_num}, bbox={[round(c) for c in table_el['bbox']]} ---")
            crop = ingestor.render_high_res_crop(args.pdf_path, page_num, table_el["bbox"])
            crop_path = os.path.join(args.out_dir, f"page{page_num}_table{i}_crop.png")
            crop.save(crop_path)
            print(f"  [crop] Saved: {crop_path} (size={crop.size})")

            raw_output = table_parser._run_qwen(
                crop, table_parser.SYSTEM_PROMPT, config.max_new_tokens_table
            )
            print(f"  [qwen-raw] {len(raw_output)} chars returned:")
            print(f"  {raw_output[:2000]!r}")

            stripped = table_parser._strip_fences(raw_output)
            split = table_parser._split_tables(stripped)
            print(f"  [split_tables] found {len(split)} table(s) after regex split")

            if not raw_output.strip():
                print(f"  [RESULT] Qwen returned EMPTY output for this crop — "
                      f"model/prompt/image issue, not a regex issue.")
            elif not split:
                print(f"  [RESULT] Qwen returned text but NO <table> tag was found by "
                      f"_split_tables — model likely didn't follow the HTML-only "
                      f"instruction. See raw output above.")
            else:
                print(f"  [RESULT] Extraction succeeded for this table.")

    ingestor.close()
    print(f"\nDone. Inspect PNGs in {args.out_dir}/ to see exactly what the detector "
          f"and model saw at each stage.")


if __name__ == "__main__":
    main()