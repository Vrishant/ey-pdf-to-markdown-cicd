# Location: src/pdf2md/observability.py
import json
import logging
import time
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


class PageMetricsRecorder:
    def __init__(self):
        self.logger = logging.getLogger("pdf2md.metrics")

    def record_page(self, page_num: int, duration_s: float, element_count: int, error: bool):
        self.logger.info(
            json.dumps({
                "event": "page_processed",
                "page_num": page_num,
                "duration_s": round(duration_s, 3),
                "element_count": element_count,
                "error": error,
            })
        )

    def record_document(self, pdf_path: str, total_duration_s: float, page_count: int):
        self.logger.info(
            json.dumps({
                "event": "document_processed",
                "pdf_path": pdf_path,
                "total_duration_s": round(total_duration_s, 3),
                "page_count": page_count,
                "avg_s_per_page": round(total_duration_s / max(page_count, 1), 3),
            })
        )


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def new_request_id() -> str:
    rid = str(uuid.uuid4())
    request_id_var.set(rid)
    return rid
