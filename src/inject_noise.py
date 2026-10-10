"""Inject controlled label errors into a clean object detection dataset.

The goal is to measure a label-error detection method against known errors. Each noisy image receives
one of four error types, and the injected errors are recorded in manifest.json:
  - missing      : a box is deleted (unlabeled object)
  - wrong_class  : a box gets a different class
  - shifted_box  : a box is displaced
  - spurious_box : a box is added where there is no object

Example:
    python src/inject_noise.py --data coco128.yaml --rate 0.2 --out data/noisy
"""

import argparse
import json
import os
import shutil
from glob import glob

import numpy as np

from utils import label_path_for

NOISE_TYPES = ["missing", "wrong_class", "shifted_box", "spurious_box"]


def parse_args():
    p = argparse.ArgumentParser(description="Controlled label noise injection")
    p.add_argument("--data", default="coco128.yaml", help="Ultralytics dataset YAML (train split is used)")
    p.add_argument("--images", default=None, help="Image folder (overrides --data if given)")
    p.add_argument("--rate", type=float, default=0.2, help="Fraction of images to corrupt")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--out", default="data/noisy")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve(data, images):
    if images:
        return images, None
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(data)
    tr = d["train"][0] if isinstance(d["train"], list) else d["train"]
    return tr, len(d["names"])


def read_rows(path):
    if not os.path.exists(path):
        return []
    return [[float(v) for v in line.split()[:5]] for line in open(path) if len(line.split()) >= 5]


def add_noise(rows, rng, n_cls):
    t = NOISE_TYPES[rng.integers(len(NOISE_TYPES))]
    i = int(rng.integers(len(rows)))
    info = {"type": t, "box": i}
    if t == "missing":
        info["removed"] = rows.pop(i)
    elif t == "wrong_class":
        old = int(rows[i][0])
        new = int(rng.choice([c for c in range(n_cls) if c != old]))
        rows[i][0] = new
        info.update(old=old, new=new)
    elif t == "shifted_box":
        c, cx, cy, w, h = rows[i]
        dx, dy = rng.choice([-1, 1]) * 0.6 * w, rng.choice([-1, 1]) * 0.6 * h
        rows[i] = [c, float(np.clip(cx + dx, w / 2, 1 - w / 2)), float(np.clip(cy + dy, h / 2, 1 - h / 2)), w, h]
    else:  # spurious_box
        w, h = rng.uniform(0.05, 0.25, size=2)
        cx, cy = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
        rows.append([int(rng.integers(n_cls)), float(cx), float(cy), float(w), float(h)])
    return info


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    img_dir, n_names = resolve(a.data, a.images)
    n_cls = a.num_classes or n_names or 80
    images = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))

    out_img, out_lbl = os.path.join(a.out, "images", "train"), os.path.join(a.out, "labels", "train")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    manifest = {}
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        shutil.copy2(img, out_img)
        rows = read_rows(label_path_for(img))
        if rows and rng.random() < a.rate:
            manifest[stem] = add_noise(rows, rng, n_cls)
        with open(os.path.join(out_lbl, stem + ".txt"), "w") as f:
            for c, cx, cy, w, h in rows:
                f.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    counts = {t: sum(v["type"] == t for v in manifest.values()) for t in NOISE_TYPES}
    print(f"Copied {len(images)} images, injected errors into {len(manifest)}: {counts}")
    print(f"Output: {a.out}  (manifest.json lists the injected errors)")


if __name__ == "__main__":
    main()
