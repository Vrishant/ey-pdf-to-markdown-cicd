"""PDF-to-Markdown extraction pipeline: vision-language-model-driven
document conversion, optimized for LLM-ready output."""

from .config import PipelineConfig
from .pipeline import PDF2MarkdownPipeline

__version__ = "0.1.0"
__all__ = ["PipelineConfig", "PDF2MarkdownPipeline"]
