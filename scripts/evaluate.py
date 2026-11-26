#!/usr/bin/env python
"""Évalue la détection (mAP) d'un modèle YOLO sur un split et sauvegarde un JSON.

python scripts/evaluate.py --model runs/train/visdrone_v8s/weights/best.pt \
    --data data/visdrone_yolo/data.yaml --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", default="outputs/metrics_detection.json")
    args = ap.parse_args()

    m = YOLO(args.model)
    r = m.val(data=args.data, split=args.split, imgsz=args.imgsz)

    metrics = {
        "model": args.model,
        "split": args.split,
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "mAP50": float(r.box.map50),
        "mAP50-95": float(r.box.map),
        "per_class_mAP50-95": {m.names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
