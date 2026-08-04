# Location: src/pdf2md/serve.py
import os
import tempfile

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse

from .config import PipelineConfig
from .pipeline import PDF2MarkdownPipeline
from .observability import configure_logging, new_request_id

configure_logging()
app = FastAPI(title="pdf2md")
_pipeline: PDF2MarkdownPipeline | None = None


@app.on_event("startup")
def load_pipeline():
    global _pipeline
    config = PipelineConfig(model_dir=os.environ.get("PDF2MD_MODEL_DIR", "/app/models"))
    _pipeline = PDF2MarkdownPipeline(config)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/convert", response_class=PlainTextResponse)
async def convert(file: UploadFile, max_pages: int | None = None):
    new_request_id()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return _pipeline.process_document(tmp_path, max_pages=max_pages)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
