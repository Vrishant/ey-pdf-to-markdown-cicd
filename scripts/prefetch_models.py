# Location: scripts/prefetch_models.py
import os

from pdf2md.model_fetch import fetch_models

MODEL_DIR = os.environ.get("PDF2MD_MODEL_DIR", "/kaggle/working/models")


def main():
    fetch_models(MODEL_DIR)


if __name__ == "__main__":
    main()