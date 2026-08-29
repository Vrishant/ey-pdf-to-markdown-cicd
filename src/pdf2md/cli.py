# Location: src/pdf2md/cli.py
import argparse
import logging
import sys
import time
from pathlib import Path

from .config import PipelineConfig
from .model_fetch import fetch_models
from .observability import configure_logging
from .pipeline import PDF2MarkdownPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the pdf2md pipeline against a local PDF")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .md path (default: <pdf_name>.md next to input)")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit to first N pages")
    parser.add_argument("--model-dir", type=str, default="/kaggle/working/models", help="Where model weights live/download to")
    parser.add_argument("--quantization-bits", type=int, choices=[4, 8], default=8)
    parser.add_argument("--no-quantization", action="store_true", help="Load full fp16 instead of quantized")
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger("pdf2md.cli")

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else pdf_path.with_suffix(".md")

    config = PipelineConfig(
        model_dir=args.model_dir,
        quantization_bits=None if args.no_quantization else args.quantization_bits,
    )

    logger.info(f"Ensuring model weights are present in {args.model_dir} ...")
    fetch_models(args.model_dir)

    logger.info(f"Loading pipeline (quantization_bits={config.quantization_bits})...")
    t0 = time.perf_counter()
    pipeline = PDF2MarkdownPipeline(config)
    logger.info(f"Pipeline ready in {time.perf_counter() - t0:.1f}s")

    logger.info(f"Processing {pdf_path} (max_pages={args.max_pages})...")
    t1 = time.perf_counter()
    markdown = pipeline.process_document(str(pdf_path), max_pages=args.max_pages)
    elapsed = time.perf_counter() - t1

    output_path.write_text(markdown, encoding="utf-8")
    logger.info(f"Done in {elapsed:.1f}s -> {output_path}")
    print(f"\nWrote {len(markdown)} chars to {output_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()