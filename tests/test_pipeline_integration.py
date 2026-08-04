# Location: tests/test_pipeline_integration.py
import os

import pytest

from pdf2md.config import PipelineConfig
from pdf2md.pipeline import PDF2MarkdownPipeline

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.mark.gpu
def test_pipeline_produces_valid_markdown():
    config = PipelineConfig()
    pipeline = PDF2MarkdownPipeline(config)
    result = pipeline.process_document(os.path.join(FIXTURE_DIR, "sample1.pdf"))

    assert result.startswith("---")
    assert "title:" in result
    assert "Sample Report Title" in result or "Sample" in result


@pytest.mark.gpu
def test_pipeline_handles_multi_page_document():
    config = PipelineConfig()
    pipeline = PDF2MarkdownPipeline(config)
    result = pipeline.process_document(os.path.join(FIXTURE_DIR, "sample2.pdf"), max_pages=1)
    assert len(result) > 0
