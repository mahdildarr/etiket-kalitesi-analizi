"""Effect of label errors on model performance: training on clean, noisy and cleaned data.

Three models are trained with identical settings and evaluated on the same clean validation set:
  1. Clean   : original labels
  2. Noisy   : labels corrupted by inject_noise.py
  3. Cleaned : the noisy data with the images flagged by find_issues.py removed

Example:
    python src/train_compare.py --data coco128.yaml --noisy data/noisy \
        --report runs/label_quality/report.csv --model yolo11n.pt --epochs 30
"""

import argparse
import csv
import os
import shutil

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Clean / noisy / cleaned training comparison")
    p.add_argument("--data", default="coco128.yaml", help="Clean dataset YAML (its val split is used for evaluation)")
    p.add_argument("--noisy", default="data/noisy", help="Output folder of inject_noise.py")
    p.add_argument("--report", default="runs/label_quality/report.csv", help="Report from find_issues.py")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work", default="data/experiment")
    return p.parse_args()


def write_yaml(path, root, train, val, names):
    cfg = {"path": os.path.abspath(root), "train": train, "val": val, "names": names}
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def build_cleaned(noisy, report, work):
    flagged = set()
    with open(report, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["flagged"] == "1":
                flagged.add(row["image"])
    root = os.path.join(work, "cleaned")
    img_out, lbl_out = os.path.join(root, "images", "train"), os.path.join(root, "labels", "train")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    kept = 0
    for fn in os.listdir(os.path.join(noisy, "images", "train")):
        stem = os.path.splitext(fn)[0]
        if stem in flagged:
            continue
        shutil.copy2(os.path.join(noisy, "images", "train", fn), img_out)
        src_lbl = os.path.join(noisy, "labels", "train", stem + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, lbl_out)
        kept += 1
    print(f"Cleaned data: {kept} images ({len(flagged)} flagged images removed)")
    return root


def main():
    a = parse_args()
    clean = check_det_dataset(a.data)
    names = clean["names"]
    val = clean["val"][0] if isinstance(clean["val"], list) else clean["val"]
    train_clean = clean["train"][0] if isinstance(clean["train"], list) else clean["train"]
    os.makedirs(a.work, exist_ok=True)

    cleaned_root = build_cleaned(a.noisy, a.report, a.work)
    setups = {
        "Clean": write_yaml(os.path.join(a.work, "clean.yaml"), os.path.dirname(train_clean), train_clean, val, names),
        "Noisy": write_yaml(os.path.join(a.work, "noisy.yaml"), a.noisy, "images/train", val, names),
        "Cleaned": write_yaml(os.path.join(a.work, "cleaned.yaml"), cleaned_root, "images/train", val, names),
    }

    results = {}
    for label, data_yaml in setups.items():
        print(f"\n=== Training: {label} ===")
        model = YOLO(a.model)
        model.train(data=data_yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=True,
                    project="runs/comparison", name=label.lower(), exist_ok=True, plots=False)
        m = model.val(data=data_yaml, imgsz=a.imgsz, project="runs/comparison", name=f"{label.lower()}_val", exist_ok=True)
        results[label] = (m.box.map50, m.box.map)

    lines = ["| Training data | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|"]
    lines += [f"| {k} | {v[0]:.4f} | {v[1]:.4f} |" for k, v in results.items()]
    text = "\n".join(lines)
    os.makedirs("runs/comparison", exist_ok=True)
    open("runs/comparison/results.md", "w", encoding="utf-8").write(text + "\n")
    print("\n" + text + "\n\nSaved: runs/comparison/results.md")


if __name__ == "__main__":
    main()
