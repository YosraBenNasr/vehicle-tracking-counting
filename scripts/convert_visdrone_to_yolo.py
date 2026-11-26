#!/usr/bin/env python
"""Convertit VisDrone-DET (annotations .txt) au format YOLO + écrit le data.yaml.

Format VisDrone, une ligne par objet :
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
Catégories : 0 = ignored regions, 1..10 = pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor, 11 = others.

Format YOLO : "classe x_centre y_centre largeur hauteur" (valeurs normalisées 0-1).

Usage
-----
python scripts/convert_visdrone_to_yolo.py --root data/VisDrone \
    --splits VisDrone2019-DET-train VisDrone2019-DET-val VisDrone2019-DET-test-dev \
    --vehicles-only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
VEHICLE_NAMES = ["car", "van", "truck", "bus", "motor"]


def convert_split(split_dir: Path, out_root: Path, split_name: str, names: list[str]) -> int:
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    out_img = out_root / "images" / split_name
    out_lbl = out_root / "labels" / split_name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_boxes = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        w, h = Image.open(img_path).size
        lines: list[str] = []
        ann = ann_dir / f"{img_path.stem}.txt"
        if ann.exists():
            for row in ann.read_text().strip().splitlines():
                parts = row.strip().split(",")
                if len(parts) < 6:
                    continue
                x, y, bw, bh = map(float, parts[:4])
                cat = int(parts[5])
                if not 1 <= cat <= 10:  # on ignore "ignored regions" (0) et "others" (11)
                    continue
                name = VISDRONE_NAMES[cat - 1]
                if name not in names:
                    continue
                cls = names.index(name)
                xc, yc = (x + bw / 2) / w, (y + bh / 2) / h
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}")
                n_boxes += 1
        (out_lbl / f"{img_path.stem}.txt").write_text("\n".join(lines))
        shutil.copy2(img_path, out_img / img_path.name)
    return n_boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="dossier contenant les splits VisDrone")
    ap.add_argument("--out", type=Path, default=Path("data/visdrone_yolo"))
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--vehicles-only", action="store_true", help="ne garde que car/van/truck/bus/motor")
    args = ap.parse_args()

    names = VEHICLE_NAMES if args.vehicles_only else VISDRONE_NAMES
    mapping = {"train": "train", "val": "val", "test": "test"}

    for s in args.splits:
        key = next((v for k, v in mapping.items() if k in s.lower()), s)
        n = convert_split(args.root / s, args.out, key, names)
        print(f"{s}: {n} boîtes -> {args.out}/images/{key}")

    yaml_text = (
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (args.out / "data.yaml").write_text(yaml_text)
    print("data.yaml écrit :", args.out / "data.yaml")


if __name__ == "__main__":
    main()
