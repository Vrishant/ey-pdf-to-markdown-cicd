# Location: scripts/prefetch_models.py
import os

from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("PDF2MD_MODEL_DIR", "/app/models")


def main():
    snapshot_download(
        repo_id="Qwen/Qwen2-VL-2B-Instruct",
        local_dir=f"{MODEL_DIR}/qwen2-vl-2b",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        local_dir=f"{MODEL_DIR}/doclayout-yolo",
        local_dir_use_symlinks=False,
    )


if __name__ == "__main__":
    main()
