# Location: scripts/prefetch_models.py
import os

from huggingface_hub import snapshot_download

# Matches PipelineConfig.model_dir default — override both consistently via
# PDF2MD_MODEL_DIR if you ever change one.
MODEL_DIR = os.environ.get("PDF2MD_MODEL_DIR", "/kaggle/working/models")


def _needs_download(path: str) -> bool:
    return not os.path.isdir(path) or len(os.listdir(path)) == 0


def main():
    qwen_dir = f"{MODEL_DIR}/qwen2-vl-2b"
    yolo_dir = f"{MODEL_DIR}/doclayout-yolo"

    if _needs_download(qwen_dir):
        print(f"Downloading Qwen2-VL-2B-Instruct to {qwen_dir} ...")
        snapshot_download(
            repo_id="Qwen/Qwen2-VL-2B-Instruct",
            local_dir=qwen_dir,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"Qwen2-VL-2B already present at {qwen_dir}, skipping.")

    if _needs_download(yolo_dir):
        print(f"Downloading DocLayout-YOLO-DocStructBench to {yolo_dir} ...")
        snapshot_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            local_dir=yolo_dir,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"DocLayout-YOLO already present at {yolo_dir}, skipping.")

    # Fail loudly rather than letting pytest hit an IndexError downstream
    pt_files = [f for f in os.listdir(yolo_dir) if f.endswith(".pt")]
    if not pt_files:
        raise RuntimeError(f"No .pt weight file found in {yolo_dir} after download.")


if __name__ == "__main__":
    main()