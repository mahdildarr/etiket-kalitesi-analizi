"""Automatically find label errors using a reference detector's predictions (cleanlab).

A 0–1 label quality score is computed for each image; a low score means the labels disagree with the
model's predictions. If manifest.json is provided, detected issues are compared against the injected
errors to measure how well the method works.

Example:
    python src/find_issues.py --images data/noisy/images/train --weights yolo11m.pt \
        --manifest data/noisy/manifest.json --save-vis 8
"""

import argparse
import csv
import json
import os
from glob import glob

import cv2
import numpy as np
from ultralytics import YOLO

from utils import label_path_for, load_yolo_labels


def parse_args():
    p = argparse.ArgumentParser(description="Label error detection with cleanlab")
    p.add_argument("--images", required=True)
    p.add_argument("--weights", default="yolo11m.pt", help="Reference model (must not have been trained on these labels)")
    p.add_argument("--conf", type=float, default=0.05, help="Low threshold: cleanlab also needs low-confidence predictions")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--manifest", default=None, help="Output of inject_noise.py (for evaluation)")
    p.add_argument("--save-vis", type=int, default=0, help="Visualize the N most suspicious images")
    p.add_argument("--out", default="runs/label_quality")
    return p.parse_args()


def collect(model, images, n_cls, conf, imgsz):
    labels, predictions = [], []
    for img in images:
        r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False, max_det=300)[0]
        h, w = r.orig_shape
        gt_cls, gt_boxes = load_yolo_labels(label_path_for(img), w, h)
        labels.append({"bboxes": gt_boxes.astype(np.float32), "labels": gt_cls.astype(int)})

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        score = r.boxes.conf.cpu().numpy()
        pred = np.empty(n_cls, dtype=object)
        for k in range(n_cls):
            m = cls == k
            pred[k] = np.hstack([xyxy[m], score[m, None]]).astype(np.float32) if m.any() else np.zeros((0, 5), np.float32)
        predictions.append(pred)
    return labels, predictions


def draw(img_path, label, pred, names, score):
    img = cv2.imread(img_path)
    gt_vis, pr_vis = img.copy(), img.copy()
    for b, c in zip(label["bboxes"].astype(int), label["labels"]):
        cv2.rectangle(gt_vis, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 2)
        cv2.putText(gt_vis, str(names.get(int(c), c)), (b[0], max(b[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for k, arr in enumerate(pred):
        for x1, y1, x2, y2, s in arr:
            if s >= 0.4:
                cv2.rectangle(pr_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                cv2.putText(pr_vis, str(names.get(k, k)), (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(gt_vis, f"Labels (score {score:.2f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(pr_vis, "Model predictions", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.hstack([gt_vis, pr_vis])


def main():
    a = parse_args()
    from cleanlab.object_detection.filter import find_label_issues
    from cleanlab.object_detection.rank import get_label_quality_scores

    images = sorted(glob(os.path.join(a.images, "*.jpg")) + glob(os.path.join(a.images, "*.png")))
    model = YOLO(a.weights)
    n_cls = len(model.names)
    print(f"Collecting predictions for {len(images)} images...")
    labels, predictions = collect(model, images, n_cls, a.conf, a.imgsz)

    scores = get_label_quality_scores(labels, predictions)
    issues = find_label_issues(labels, predictions)
    order = np.argsort(scores)

    manifest = json.load(open(a.manifest, encoding="utf-8")) if a.manifest else {}
    os.makedirs(a.out, exist_ok=True)
    stems = [os.path.splitext(os.path.basename(p))[0] for p in images]
    with open(os.path.join(a.out, "report.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "image", "quality_score", "flagged", "num_labels", "injected_error"])
        for rank, i in enumerate(order, start=1):
            wr.writerow([rank, stems[i], f"{scores[i]:.4f}", int(issues[i]), len(labels[i]["labels"]),
                         manifest.get(stems[i], {}).get("type", "")])

    lines = [f"Images: {len(images)}", f"Flagged as issues: {int(issues.sum())}"]
    if manifest:
        noisy = np.array([s in manifest for s in stems])
        tp = int((issues & noisy).sum())
        k = int(noisy.sum())
        top_k = np.zeros(len(images), dtype=bool)
        top_k[order[:k]] = True
        lines += [
            f"Images with injected errors: {k}",
            f"Precision (share of flagged images that are truly corrupted): {tp / max(issues.sum(), 1):.3f}",
            f"Recall (share of injected errors that were found): {tp / max(k, 1):.3f}",
            f"Precision@{k} (the {k} lowest-scoring images): {(top_k & noisy).sum() / max(k, 1):.3f}",
            "",
            "Recall by error type:",
        ]
        for t in ["missing", "wrong_class", "shifted_box", "spurious_box"]:
            idx = [i for i, s in enumerate(stems) if manifest.get(s, {}).get("type") == t]
            if idx:
                lines.append(f"  {t:<13}: {issues[idx].mean():.3f}  ({len(idx)} images)")
    text = "\n".join(lines)
    open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8").write(text + "\n")
    print("\n" + text)

    if a.save_vis:
        vis_dir = os.path.join(a.out, "most_suspicious")
        os.makedirs(vis_dir, exist_ok=True)
        for i in order[: a.save_vis]:
            cv2.imwrite(os.path.join(vis_dir, f"{stems[i]}.jpg"), draw(images[i], labels[i], predictions[i], model.names, scores[i]))
    print(f"\nReport: {a.out}/report.csv  ·  Summary: {a.out}/summary.txt")


if __name__ == "__main__":
    main()
