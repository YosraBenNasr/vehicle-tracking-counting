#!/usr/bin/env python
"""Fine-tuning de YOLO sur un jeu de données au format YOLO (VisDrone converti, UA-DETRAC...).

Exemple
-------
python scripts/train.py --data data/visdrone_yolo/data.yaml --model yolov8s.pt \
    --epochs 50 --imgsz 960 --batch 8 --name visdrone_v8s
Si "CUDA out of memory" : réduire --batch ou --imgsz (voir docs/TROUBLESHOOTING.md).
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15, help="arrêt anticipé (early stopping)")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        # augmentations adaptées à la vue de dessus / trafic
        fliplr=0.5,
        mosaic=1.0,
        hsv_v=0.4,
    )


if __name__ == "__main__":
    main()
