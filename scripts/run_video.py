#!/usr/bin/env python
"""Lance le pipeline sur une vidéo ou une webcam.

Exemples
--------
python scripts/run_video.py --source data/videos/traffic.mp4 --config configs/default.yaml
python scripts/run_video.py --source 0 --show                     # webcam
python scripts/run_video.py --source traffic.mp4 --speed          # active la vitesse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from vtc.config import load_config
from vtc.pipeline import run_video, save_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="chemin vidéo, URL de flux, ou index de webcam (0)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", help="remplace detector.model (ex. yolo11s.pt ou runs/.../best.pt)")
    ap.add_argument("--conf", type=float, help="remplace detector.conf")
    ap.add_argument("--speed", action="store_true", help="active l'estimation de vitesse")
    ap.add_argument("--show", action="store_true", help="affiche la fenêtre en direct (q pour quitter)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.detector.model = args.model
    if args.conf is not None:
        cfg.detector.conf = args.conf
    if args.speed:
        cfg.speed.enabled = True
    if args.out_dir:
        cfg.output.out_dir = args.out_dir

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    stem = "webcam" if isinstance(source, int) else Path(source).stem
    out_dir = Path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    pipe = None
    for pipe, res in run_video(source, cfg, max_frames=args.max_frames):
        if cfg.output.save_video and writer is None:
            h, w = res.frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_dir / f"{stem}_annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                pipe.fps,
                (w, h),
            )
        if writer is not None:
            writer.write(res.frame)
        for ev in res.new_events:
            sp = "" if ev.speed_kmh is None else f" @ {ev.speed_kmh:.0f} km/h"
            print(f"[t={ev.timestamp_s:6.1f}s] #{ev.track_id} {ev.class_name} -> {ev.direction}{sp}")
        if args.show:
            cv2.imshow("vtc", res.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if pipe is not None:
        if cfg.output.save_events_csv:
            print("Événements enregistrés :", save_outputs(pipe, out_dir, stem))
        print("Comptage final :", pipe.counter.as_dict(), "| total =", pipe.counter.total)


if __name__ == "__main__":
    main()
