# pdf2md — Changes & Iteration Log

Chronological record of every design decision and change made to this project, from initial notebook review through the current CLI/CI-CD state. Grouped by phase in the order they actually happened.

---

## Phase 0 — Source material review
Reviewed three independent Kaggle prototype notebooks before writing any merged code:
- **`qwen-2b-table-optimisation.ipynb`** — table-extraction tuning: box-merging heuristic, greedy-decoding findings, YOLO confidence threshold sweeps, token-level F1 evaluation harness. Single-page only, no document assembly.
- **`pdf2md_v2.ipynb`** — modular OOP architecture (Ingestor/Extractor/Router/Parsers), two-column reading-order sorter, bounding-box debug visualizer. Used blind OCR even on born-digital text.
- **`pdf2md_v3.ipynb`** — dual-resolution rendering (cheap layout pass + high-DPI extraction crop), strict multi-table prompt, isolated per-table output. Tables only — no text/figures/headings.

## Phase 1 — Initial 11-section merged pipeline (HTML output)
Built section by section: dual-resolution `PDFIngestor`, `LayoutExtractor` with **exact-match** class mapping (fixing a real bug in v2/v3 where `table_caption` was substring-matched into `table`), noise filtering, type-aware box merging, reading-order sort, caption association, metadata extraction, native-PDF-text `TextHeadingParser` (replacing blind OCR), `TableParser`, `FigureParser` (base64-embedded images + Qwen caption), `ContentRouter`, and an `HTMLDocumentAssembler` producing a full styled HTML5 document with CSS.

## Phase 2 — Output format pivot #1: token-optimized HTML
Discovered the generated `output.html` was ~96% base64 image bytes (1,019,224 of 1,061,118 bytes on a real sample). Removed `<img>` embedding entirely, dropped presentation CSS, compacted metadata into a single JSON block instead of scattered `<meta>` tags.

## Phase 3 — Output format pivot #2: full Markdown
Replaced `HTMLDocumentAssembler` with `MarkdownDocumentAssembler` (YAML frontmatter + per-page HTML-comment sections). `TextHeadingParser` now emits `#`-style headings. Tables kept as embedded HTML `<table>` (for rowspan/colspan fidelity) inside the `.md` file rather than converting to lossy GFM pipe tables.

## Phase 4 — Figure handling evolution
Went through three distinct designs in response to changing requirements:
1. `FigureParser` — embed cropped image as base64 + Qwen-generated caption.
2. `GraphParser` v1 — classify each figure as `DATA` vs `DECORATIVE`; `DATA` figures get their values extracted as a table, `DECORATIVE` ones get **dropped**.
3. `GraphParser` v2 (final, current) — same classification, but `DECORATIVE` figures now get a text description instead of being dropped — nothing is silently discarded. Later optimized (Phase 8) to do classification and extraction/description in a **single** combined Qwen call instead of two.

## Phase 5 — Table isolation fix
Found that `merge_neighboring_boxes` could fuse two visually adjacent but distinct tables into one bounding box, mixing their content in one Qwen call. Fixed by restricting merging to `text`-type boxes only — tables and figures are never merged. Added `TableParser._split_tables` to also split multiple `<table>` blocks that appear within a single crop, so each table is always an isolated output.

## Phase 6 — Quantization
Added 8-bit (`bitsandbytes`) quantized loading for Qwen2-VL-2B via `BitsAndBytesConfig`, configurable through `PipelineConfig.quantization_bits` (`8`, `4`, or `None` for full fp16).

## Phase 7 — Optimization & CI/CD planning
Produced a phased plan: **A0/A1** (speed quick-wins, no accuracy risk), **A2/A3** (batching, serving-layer changes), **B0–B4** (package refactor, CI, containerization/CD, deployment shape, monitoring). Sequencing: B0 first, since CI is meaningless against raw notebook cells.

## Phase 8 — B0: package refactor
Split the notebook into an installable `src/pdf2md/` package: `config.py`, `ingestion.py` (also fixed **A0** here — `PDFIngestor` now caches one open `fitz.Document` per path instead of reopening the file on every single element), `layout.py`, `ordering.py`, `metadata.py`, `parsers/{text,table,graph}.py`, `router.py` (also applied **A1**: `clear_vram()` throttled to every 5 VLM calls instead of every element, `GraphParser` merged into one combined classify+extract/describe Qwen call), `assembler.py`, `pipeline.py`.

## Phase 9 — B1/B2/B3/B4: CI, containerization, serving, observability
- `ci.yml` — lint + fast unit tests (no GPU) on every PR/push to `main`.
- `nightly.yml` — scheduled full test suite + F1 regression gate on a self-hosted GPU runner.
- `Dockerfile` + `scripts/prefetch_models.py` — multi-stage build, optional weight-baking.
- `src/pdf2md/serve.py` — FastAPI `/convert` endpoint.
- `cd.yml` — build/push image, auto-deploy staging, gated production deploy.
- `src/pdf2md/observability.py` — structured JSON logging, per-page/per-document latency metrics.

## Phase 10 — Kaggle-hosted self-hosted GPU runner
Registered a Kaggle notebook as an ephemeral GitHub Actions runner (`self-hosted, gpu, kaggle` labels) via `notebooks/kaggle_gpu_runner_cell.py`. Debugged along the way:
- Hardcoded runner version 404'd — fixed by resolving the latest release URL dynamically via the GitHub API.
- `Must not run with sudo` — fixed with `RUNNER_ALLOW_RUNASROOT=1` (Kaggle runs as root).
- Established that scheduled jobs **queue** for offline self-hosted runners rather than failing, so periodic Kaggle notebook scheduling (not true 24/7 uptime) is sufficient.

## Phase 11 — Automatic GPU testing on PR
Added `gpu-pr-tests.yml`, triggered on PRs from `develop` into `main`, gated behind a `gpu-tests` GitHub Environment requiring manual approval (deliberate safety gate — self-hosted runner + PR-triggered code execution is a real risk even for same-repo PRs).

## Phase 12 — CI/CD fixes session (see `CICD_FIXES.md` for full detail)
1. **GPU tests failing with `IndexError`** — `prefetch_models.py` was never called in `gpu-pr-tests.yml`, and its default `MODEL_DIR` (`/app/models`) didn't match `PipelineConfig`'s default (`/kaggle/working/models`). Fixed both.
2. **Ruff import-order failures (`I001`)** — reordered imports across `config.py`, `layout.py`, `parsers/__init__.py`, `graph.py`, `table.py`, `pipeline.py`, `router.py`, `serve.py`, tests.
3. **Black blocking merges on a Python-version mismatch** — made lint/format `continue-on-error: true`; pinned `[tool.black] target-version = ["py311"]`.
4. **CD build failing on uppercase repo name** — Docker requires lowercase image names; `github.repository` included a capital letter. Fixed by computing a lowercased image name at runtime instead of a static `env.IMAGE_NAME`.

## Phase 13 — A1 completion: dynamic token budgeting + CLI
- `src/pdf2md/utils.py::estimate_max_tokens` — scales `max_new_tokens` by crop area instead of always requesting the full 1500-token budget regardless of table size.
- `src/pdf2md/cli.py` — `pdf2md` console-script entry point for manual testing against arbitrary PDFs (`pip install -e .` then `pdf2md file.pdf`).

## Phase 14 — CLI bugfix: model weights never fetched
CLI raised `IndexError: list index out of range` in `LayoutExtractor` — no `.pt` file found, because `prefetch_models.py`'s download logic was never called from the CLI path (only from CI workflows). Fixed by extracting the shared logic into `src/pdf2md/model_fetch.py::fetch_models()`, called automatically at the top of `cli.py::main()`; `scripts/prefetch_models.py` is now a thin wrapper around the same function, closing the duplication gap that caused this.

## Phase 15 — Large-table OOM mitigation
Kaggle GPU ran out of memory on large tables. Added band-splitting to `TableParser`: crops taller than `table_split_threshold_px` are sliced into overlapping vertical PDF-space bands (`table_split_band_pt`, `table_split_overlap_pt`), each sent to Qwen independently with a prompt to omit rows visibly cut at the image edge, then merged with consecutive-duplicate-row dedup and `thead`/`tbody` reassembly. **Known limitation**: a `rowspan`/`colspan` cell straddling a band boundary won't reconstruct correctly.

## Phase 16 — 🔴 Open issue: scanned-PDF real-world test shows zero table extraction
First real test against `input.pdf` (a **scanned** multi-page financial statement, not born-digital) + `finance_multiheader_scan_ground_truth.csv` produced an `output.md` containing only page titles and currency-unit fragments (`(® in Lakhs)`) — **no table content extracted on any of the 3 pages**, despite each page clearly containing a dense financial table.

Suspected causes, not yet root-caused:
- Since the source is scanned (image-only), `get_native_text` correctly returns nothing for body text — expected — but the **layout detector may not be finding table regions at all** on this scan quality/style, or table crops are being sent but Qwen is returning empty/unparseable output that `_split_tables` then discards.
- Worth checking next: run `DocumentVisualizer`-equivalent bounding-box overlay on this specific PDF to see whether `LayoutExtractor` is detecting `table` boxes at all before assuming the extraction prompt is at fault.

**Status: unresolved, next debugging priority.**

## Phase 17 — GPU acceleration & Flash Attention
Qwen2-VL was running predominantly on CPU with minimal GPU utilization, making table extraction extremely slow. Fixed by:
- Forcing `device_map={"":"cuda"}` instead of `"auto"` to place the entire model on the GPU.
- Enabling `attn_implementation="sdpa"` (Scaled Dot-Product Attention / Flash Attention) for faster attention kernels.
- Using `torch.float16` precision unconditionally (not just when quantization is off).
These changes together moved all inference to GPU and eliminated the CPU bottleneck.

## Phase 18 — Header-stitching table splitter (complete `TableParser` rewrite)
The Phase 15 band-splitting approach had a critical flaw: when a large table was sliced into horizontal bands, the lower bands lost the column headers entirely, so Qwen2-VL couldn't align data to columns — producing jumbled or hallucinated output on dense multi-column financial tables.

**Rewrite summary — new flow:**
1. YOLO detects table → `render_high_res_crop` produces a single PIL image of the full table.
2. If `crop.height <= table_split_threshold_px` (800px), process in one shot with `FULL_TABLE_PROMPT` (unchanged fast path).
3. If the crop is taller, enter `_parse_large_table`:
   - **Header extraction**: crop the top 15% (80–400px) of the image. Send it to Qwen once with a dedicated `HEADER_PROMPT` to get `<tr><th>…</th></tr>` rows.
   - **Body slicing**: slice the remainder into horizontal strips of `table_split_threshold_px` height, with `table_split_overlap_pt`-scaled pixel overlap.
   - **Header stitching**: for each strip, use `PIL.Image` to paste the header image on top of the strip, producing a stitched image where Qwen always sees the column headers.
   - **Strip processing**: send each stitched image to Qwen with `STRIP_PROMPT` (instructs the model to extract only data rows, not re-extract the headers).
   - **Dedup & assemble**: consecutive-duplicate rows from overlap are dropped; header HTML and body rows are wrapped into `<table><thead>…</thead><tbody>…</tbody></table>`.

**Removed:** `_compute_bands` (PDF-point-space slicing), `_merge_rows` (replaced by inline dedup), `BAND_PROMPT`, `SYSTEM_PROMPT` (replaced by `FULL_TABLE_PROMPT`). Old commented-out original `TableParser` class also removed.

**Config changes:**
- `table_split_threshold_px`: 1800 → **800** — the user's real-world table crops were 1380–1677px tall, which never triggered splitting at 1800.
- `qwen_max_pixels`: 8,192,000 → **2,000,000** — prevents the processor from upscaling images to unreasonable sizes.
- `table_split_band_pt`: 1500 → **400** — restored to original intent.
