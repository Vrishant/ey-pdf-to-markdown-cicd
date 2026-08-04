# pdf2md — PDF-to-Markdown Extraction Pipeline

Vision-language-model-driven pipeline that converts PDF documents into clean, structured Markdown optimized for LLM ingestion (RAG, information extraction, document QA).

Built on **Qwen2-VL-2B-Instruct** (table/chart extraction, 8-bit quantized) and **DocLayout-YOLO** (layout detection), with native PDF text extraction for body content.

## Repository structure

```
├── src/pdf2md/            # installable package
│   ├── config.py           # PipelineConfig — single source of truth
│   ├── ingestion.py         # dual-resolution PDF rendering
│   ├── layout.py            # layout detection, class mapping, box merging
│   ├── ordering.py           # reading order, caption association
│   ├── metadata.py            # PDF metadata extraction
│   ├── parsers/                # text/heading, table, graph parsers
│   ├── router.py                # content routing per element type
│   ├── assembler.py              # Markdown + YAML frontmatter assembly
│   ├── pipeline.py                # PDF2MarkdownPipeline orchestrator
│   ├── observability.py            # structured logging + metrics
│   └── serve.py                     # FastAPI serving entrypoint
├── tests/
│   ├── fixtures/            # sample PDFs, ground truth, F1 baseline
│   ├── test_*.py             # fast unit tests (no GPU)
│   └── test_pipeline_integration.py  # @pytest.mark.gpu, full pipeline
├── scripts/
│   ├── prefetch_models.py   # bakes model weights into Docker image
│   └── check_f1_regression.py  # nightly regression gate
├── .github/workflows/
│   ├── ci.yml                # lint + fast unit tests on every PR
│   ├── nightly.yml            # GPU integration tests + F1 regression
│   └── cd.yml                  # build/push image, staged deployment
├── Dockerfile
└── pyproject.toml
```

## Getting started

```bash
pip install -e ".[dev]"
apt-get install -y tesseract-ocr   # OCR fallback for scanned pages
```

Models are pulled on first pipeline run via `huggingface_hub.snapshot_download`:
- `Qwen/Qwen2-VL-2B-Instruct`
- `juliozhao/DocLayout-YOLO-DocStructBench`

## Usage

```python
from pdf2md.config import PipelineConfig
from pdf2md.pipeline import PDF2MarkdownPipeline

pipeline = PDF2MarkdownPipeline(PipelineConfig())
markdown_output = pipeline.process_document("document.pdf", max_pages=None)

with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown_output)
```

Or via the served API (`docker build` / `python -m pdf2md.serve`):

```bash
curl -X POST http://localhost:8080/convert -F "file=@document.pdf"
```

## Testing

```bash
pytest -m "not gpu"     # fast, runs in every PR (no models/GPU needed)
pytest                   # full suite, requires GPU + downloaded models
```

## CI/CD

- **`ci.yml`** — lint (`ruff`), format check (`black`), fast unit tests on every PR and push to `main`.
- **`nightly.yml`** — scheduled GPU run of the full test suite plus an F1 regression check against `tests/fixtures/f1_baseline.json`; fails if quality drops beyond threshold.
- **`cd.yml`** — builds and pushes a versioned Docker image on merge to `main`, auto-deploys to staging, and gates production behind manual approval (GitHub Environments).

`nightly.yml` requires a self-hosted GPU runner (`runs-on: [self-hosted, gpu]`) — configure one under repo Settings → Actions → Runners before enabling it.

## Configuration

All tunables live in `PipelineConfig` (`src/pdf2md/config.py`):

| Parameter | Purpose |
|---|---|
| `layout_dpi` / `extraction_dpi` | Resolution for layout detection vs. content extraction |
| `yolo_conf_threshold` | Layout detector confidence cutoff |
| `quantization_bits` | `8`, `4`, or `None` (full precision) |
| `ocr_fallback_enabled` | Whether to OCR pages with no extractable text layer |
| `do_sample` / `temperature` | Decoding strategy for the extraction model (greedy by default) |

## License & third-party models

This repository's code is for internal EY use — update this section with your organization's licensing terms before external distribution. Third-party models (Qwen2-VL-2B-Instruct, DocLayout-YOLO-DocStructBench) are subject to their own licenses; check their Hugging Face model cards before production or client-facing use.
