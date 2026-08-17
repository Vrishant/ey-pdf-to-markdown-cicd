# pdf2md — System Flowchart

Mermaid diagrams render natively on GitHub — no extra tooling needed to view this file in the repo.

## 1. End-to-end pipeline (per document)

```mermaid
flowchart TD
    A[PDF file] --> B["PDFIngestor: render_layout_canvas 72 DPI"]
    B --> C["LayoutExtractor: DocLayout-YOLO segment_page"]
    C --> D["NoiseFilter: drop sub-threshold boxes"]
    D --> E["merge_neighboring_boxes: text only"]
    E --> F["CaptionAssociator: attach captions/footnotes to nearest table or figure"]
    F --> G["ReadingOrderSorter: column-aware sort"]
    G --> H["MarkdownDocumentAssembler.rerank_headings_for_page: font-size to h1..h6"]
    H --> I["ContentRouter.orchestrate: one element at a time"]

    I -->|text or heading| J["TextHeadingParser: native PDF text, OCR fallback if empty"]
    I -->|table| K["TableParser.parse"]
    I -->|figure| L["GraphParser.parse: single combined Qwen call"]
    I -->|orphaned caption/footnote| J

    K --> K1{"crop height > table_split_threshold_px?"}
    K1 -->|no| K2["Qwen2-VL: full-table HTML prompt"]
    K1 -->|yes| K3["compute_bands: vertical PDF-space slices with overlap"]
    K3 --> K4["Qwen2-VL per band: bare tr rows only, omit cut-off edge rows"]
    K4 --> K5["merge_rows: dedupe overlap, split thead/tbody, wrap table"]
    K2 --> K6["split_tables: multiple tables in one crop become separate outputs"]

    L --> L1{"DATA or DECORATIVE?"}
    L1 -->|DATA| L2["HTML table of extracted values"]
    L1 -->|DECORATIVE| L3["1-3 sentence description"]

    J --> M[Markdown block]
    K5 --> M
    K6 --> M
    L2 --> M
    L3 --> M

    M --> N["ContentRouter._wrap: attach associated captions/footnotes"]
    N --> O["pages_md list, one entry per page"]
    O --> P["MarkdownDocumentAssembler.assemble: YAML frontmatter + page sections"]
    P --> Q[output.md]
```

## 2. Table parser decision detail

```mermaid
flowchart LR
    A[Table element bbox] --> B["render_high_res_crop: 300 DPI"]
    B --> C{"crop.height <= table_split_threshold_px"}
    C -->|yes, normal path| D["single Qwen call, full SYSTEM_PROMPT"]
    C -->|no, oversized| E["_compute_bands: 400pt bands, 30pt overlap"]
    E --> F["one Qwen call per band, BAND_PROMPT"]
    F --> G["collect bare tr rows, in order"]
    G --> H["dedupe consecutive identical rows from overlap"]
    H --> I["separate th rows into thead, rest into tbody"]
    I --> J[merged table HTML]
    D --> K["_split_tables: regex-split multiple table blocks"]
    J --> L[List of table HTML strings]
    K --> L
```

## 3. CLI manual-test flow

```mermaid
flowchart TD
    A["pdf2md your_file.pdf"] --> B["PipelineConfig from CLI args"]
    B --> C["fetch_models: idempotent snapshot_download"]
    C --> D{".pt file present after download?"}
    D -->|no| E["raise RuntimeError"]
    D -->|yes| F["PDF2MarkdownPipeline constructed"]
    F --> G["process_document: loop over pages"]
    G --> H["per-page metrics recorded, JSON logs"]
    H --> I["write output.md"]
```

## 4. CI/CD trigger map

```mermaid
flowchart TD
    subgraph PR["Pull Request"]
        P1["push to any branch, PR opened to main"] --> P2["ci.yml: lint non-blocking, fast unit tests"]
        P3["PR opened: develop into main"] --> P4["gpu-pr-tests.yml: gated by gpu-tests Environment approval"]
        P4 --> P5["Kaggle self-hosted runner picks up queued job"]
        P5 --> P6["prefetch_models -> full pytest -> F1 regression check"]
    end

    subgraph Merge["Merge to main"]
        M1["push to main"] --> M2["cd.yml: build + push Docker image"]
        M2 --> M3["auto-deploy staging"]
        M3 --> M4{"manual dispatch: deploy_to_production?"}
        M4 -->|yes| M5["production Environment, requires approval"]
    end

    subgraph Scheduled["Scheduled"]
        S1["nightly.yml cron 03:00 or manual dispatch"] --> S2["same GPU test + F1 gate, independent of PR state"]
    end
```

## 5. Kaggle GPU runner lifecycle

```mermaid
flowchart LR
    A["Kaggle notebook: kaggle_gpu_runner_cell.py"] --> B["fetch GH_RUNNER_TOKEN from Kaggle Secrets"]
    B --> C["resolve latest actions-runner release via GitHub API"]
    C --> D["config.sh: register as self-hosted, gpu, kaggle"]
    D --> E["RUNNER_ALLOW_RUNASROOT=1 set, since Kaggle runs as root"]
    E --> F["run.sh: listening for jobs"]
    F --> G{"queued job matching labels?"}
    G -->|yes| H["run one job"]
    H --> I["--ephemeral: deregister and exit"]
    G -->|no, session ends| I
    I --> J["Kaggle Schedule triggers next run per configured cadence"]
    J --> A
```
