# Location: src/pdf2md/model_fetch.py
import logging
import os

from huggingface_hub import snapshot_download

logger = logging.getLogger("pdf2md.model_fetch")


def _needs_download(path: str) -> bool:
    return not os.path.isdir(path) or len(os.listdir(path)) == 0


def fetch_models(model_dir: str) -> None:
    qwen_dir = f"{model_dir}/qwen2-vl-2b"
    yolo_dir = f"{model_dir}/doclayout-yolo"

    if _needs_download(qwen_dir):
        logger.info(f"Downloading Qwen2-VL-2B-Instruct to {qwen_dir} ...")
        snapshot_download(
            repo_id="Qwen/Qwen2-VL-2B-Instruct",
            local_dir=qwen_dir,
            local_dir_use_symlinks=False,
        )
    else:
        logger.info(f"Qwen2-VL-2B-Instruct already present at {qwen_dir}, skipping download.")

    if _needs_download(yolo_dir):
        logger.info(f"Downloading DocLayout-YOLO-DocStructBench to {yolo_dir} ...")
        snapshot_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            local_dir=yolo_dir,
            local_dir_use_symlinks=False,
        )
    else:
        logger.info(f"DocLayout-YOLO already present at {yolo_dir}, skipping download.")

    pt_files = [f for f in os.listdir(yolo_dir) if f.endswith(".pt")]
    if not pt_files:
        raise RuntimeError(
            f"No .pt weight file found in {yolo_dir} after download — "
            "download likely failed or landed in the wrong directory."
        )