import argparse
import json
import sys
from pathlib import Path

from pdf2md.config import PipelineConfig
from pdf2md.pipeline import PDF2MarkdownPipeline

BASELINE_PATH = Path("tests/fixtures/f1_baseline.json")
FIXTURES = [
    ("tests/fixtures/sample1.pdf", "tests/fixtures/sample1_gt.xml"),
    ("tests/fixtures/sample2.pdf", "tests/fixtures/sample2_gt.xml"),
]


def tokenize(text: str):
    import re
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"^---[\s\S]*?---", " ", text)
    text = re.sub(r"[#*_`|>:+\-]+", " ", text)
    return re.findall(r"\b\w+\b", text)


def compute_f1(pred: str, gt: str):
    p, g = tokenize(pred), tokenize(gt)
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    precision = len(common) / len(set(p))
    recall = len(common) / len(set(g))
    return 2 * precision * recall / (precision + recall)


def load_ground_truth(xml_path: str) -> str:
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    return " ".join(e.text.strip() for e in root.iter() if e.text and e.text.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=-0.02)
    args = parser.parse_args()

    pipeline = PDF2MarkdownPipeline(PipelineConfig())
    scores = []
    for pdf_path, gt_path in FIXTURES:
        pred = pipeline.process_document(pdf_path)
        gt = load_ground_truth(gt_path)
        scores.append(compute_f1(pred, gt))

    current_f1 = sum(scores) / len(scores)
    baseline_f1 = json.loads(BASELINE_PATH.read_text())["f1"] if BASELINE_PATH.exists() else current_f1
    delta = current_f1 - baseline_f1

    report = {"current_f1": current_f1, "baseline_f1": baseline_f1, "delta": delta}
    Path("f1_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if delta < args.threshold:
        print(f"F1 regression exceeded threshold: {delta:.4f} < {args.threshold}")
        sys.exit(1)


if __name__ == "__main__":
    main()
