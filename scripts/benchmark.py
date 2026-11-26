#!/usr/bin/env python
"""Mesure le débit (FPS) du pipeline complet pour plusieurs modèles / résolutions.

python scripts/benchmark.py --source data/videos/traffic.mp4 \
    --models yolov8n.pt yolov8s.pt yolo11n.pt --imgsz 640 960 --frames 300
Le tableau obtenu (CSV) sert à justifier le compromis précision / vitesse.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from vtc.config import load_config
from vtc.pipeline import run_video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="outputs/benchmark.csv")
    args = ap.parse_args()

    rows = []
    for model in args.models:
        for sz in args.imgsz:
            cfg = load_config(args.config)
            cfg.detector.model, cfg.detector.imgsz = model, sz
            n, t0, pipe = 0, None, None
            for pipe, _ in run_video(args.source, cfg, max_frames=args.frames + 10):
                n += 1
                if n == 10:  # on ignore l'échauffement (chargement, cuDNN...)
                    t0 = time.perf_counter()
            elapsed = time.perf_counter() - (t0 or time.perf_counter())
            fps = (n - 10) / elapsed if elapsed > 0 else float("nan")
            rows.append(dict(model=model, imgsz=sz, fps=round(fps, 1), total_counted=pipe.counter.total if pipe else 0))
            print(rows[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
