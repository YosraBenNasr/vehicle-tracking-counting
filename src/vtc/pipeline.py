"""Pipeline complet : détection -> suivi -> comptage -> vitesse -> annotation."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .counter import CrossingEvent, LineCounter
from .detector import Track, TrackerBackend, YoloByteTrack
from .speed import SpeedEstimator

PALETTE = [
    (66, 135, 245), (245, 130, 48), (60, 180, 75), (230, 25, 75),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
]


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_s: float
    frame: np.ndarray  # image annotée (BGR)
    new_events: list[CrossingEvent]
    active_tracks: int
    counts: dict[str, dict[str, int]]
    totals: dict[str, int]
    mean_speed_kmh: float | None
    fps_processing: float


@dataclass
class Pipeline:
    cfg: Config
    detector: TrackerBackend
    fps: float
    size: tuple[int, int]  # (largeur, hauteur)
    counter: LineCounter = field(init=False)
    speed: SpeedEstimator | None = field(init=False, default=None)
    events: list[CrossingEvent] = field(default_factory=list)
    _trails: dict[int, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=30)))

    def __post_init__(self) -> None:
        w, h = self.size
        (x1, y1), (x2, y2) = self.cfg.counting.line
        self.counter = LineCounter(
            (x1 * w, y1 * h),
            (x2 * w, y2 * h),
            labels=tuple(self.cfg.counting.labels),  # type: ignore[arg-type]
            count_once=self.cfg.counting.count_once,
        )
        sp = self.cfg.speed
        if sp.enabled:
            pts = [(px * w, py * h) for px, py in sp.src_points]
            self.speed = SpeedEstimator(
                pts, sp.width_m, sp.length_m, self.fps, window_s=sp.window_s, max_kmh=sp.max_kmh
            )
        self._trails = defaultdict(lambda: deque(maxlen=self.cfg.output.trail_length))
        self._t_prev = time.perf_counter()
        self._fps_ema = 0.0

    # ------------------------------------------------------------------ utils
    def _anchor(self, t: Track) -> tuple[float, float]:
        x1, y1, x2, y2 = t.xyxy
        if self.cfg.counting.anchor == "center":
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return ((x1 + x2) / 2, y2)  # point de contact avec la route

    @staticmethod
    def _color(cls_id: int) -> tuple[int, int, int]:
        return PALETTE[cls_id % len(PALETTE)]

    # ------------------------------------------------------------ traitement
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        ts = frame_idx / self.fps
        tracks = self.detector.track(frame)
        new_events: list[CrossingEvent] = []
        speeds: list[float] = []

        for t in tracks:
            pos = self._anchor(t)
            self._trails[t.track_id].append((int(pos[0]), int(pos[1])))
            v = self.speed.update(t.track_id, pos, frame_idx) if self.speed else None
            if v is not None:
                speeds.append(v)
            ev = self.counter.update(t.track_id, pos, t.cls_name, frame_idx, ts)
            if ev is not None:
                ev.speed_kmh = v
                new_events.append(ev)
                self.events.append(ev)

        active = {t.track_id for t in tracks}
        if self.speed:
            self.speed.drop(active)
        for tid in list(self._trails):
            if tid not in active and len(self._trails) > 200:
                self._trails.pop(tid, None)

        now = time.perf_counter()
        inst = 1.0 / max(now - self._t_prev, 1e-6)
        self._t_prev = now
        self._fps_ema = inst if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst

        annotated = self.annotate(frame, tracks)
        return FrameResult(
            frame_idx=frame_idx,
            timestamp_s=ts,
            frame=annotated,
            new_events=new_events,
            active_tracks=len(tracks),
            counts=self.counter.as_dict(),
            totals=self.counter.totals_by_direction(),
            mean_speed_kmh=float(np.mean(speeds)) if speeds else None,
            fps_processing=self._fps_ema,
        )

    # ------------------------------------------------------------ annotation
    def annotate(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        out = frame.copy()

        # zone de calibration de vitesse
        if self.speed is not None:
            cv2.polylines(out, [self.speed.polygon.astype(np.int32)], True, (0, 200, 255), 1)

        # ligne de comptage
        a = tuple(int(v) for v in self.counter.a)
        b = tuple(int(v) for v in self.counter.b)
        cv2.line(out, a, b, (0, 0, 255), 3)

        for t in tracks:
            x1, y1, x2, y2 = map(int, t.xyxy)
            color = self._color(t.cls_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            pts = list(self._trails[t.track_id])
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(out, p, q, color, 2)

            label = f"#{t.track_id} {t.cls_name}"
            v = self.speed.get(t.track_id) if self.speed else None
            if v is not None:
                label += f" {v:.0f} km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # bandeau des compteurs
        tot = self.counter.totals_by_direction()
        l0, l1 = self.cfg.counting.labels
        banner = f"{l0}: {tot[l0]}   {l1}: {tot[l1]}   total: {self.counter.total}   {self._fps_ema:4.1f} FPS"
        cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(out, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    # ---------------------------------------------------------------- export
    def events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    track_id=e.track_id,
                    class_name=e.class_name,
                    direction=e.direction,
                    frame=e.frame_idx,
                    time_s=round(e.timestamp_s, 3),
                    speed_kmh=None if e.speed_kmh is None else round(e.speed_kmh, 1),
                )
                for e in self.events
            ],
            columns=["track_id", "class_name", "direction", "frame", "time_s", "speed_kmh"],
        )


# ---------------------------------------------------------------------------
def build_pipeline(cfg: Config, fps: float, size: tuple[int, int], detector: TrackerBackend | None = None) -> Pipeline:
    if detector is None:
        d = cfg.detector
        detector = YoloByteTrack(
            model=d.model, tracker=d.tracker, conf=d.conf, iou=d.iou,
            imgsz=d.imgsz, device=d.device, vehicle_classes=d.vehicle_classes,
        )
    return Pipeline(cfg=cfg, detector=detector, fps=fps, size=size)


def run_video(
    source: str | int,
    cfg: Config,
    detector: TrackerBackend | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[Pipeline, FrameResult]]:
    """Générateur : lit la vidéo/le flux et produit un FrameResult par image.

    Utilisé à la fois par la ligne de commande (scripts/run_video.py) et par le
    tableau de bord Streamlit, ce qui garantit des résultats identiques.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la source vidéo : {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipe = build_pipeline(cfg, fps, (w, h), detector)

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and idx >= max_frames):
                break
            yield pipe, pipe.process_frame(frame, idx)
            idx += 1
    finally:
        cap.release()


def save_outputs(pipe: Pipeline, out_dir: str | Path, stem: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{stem}_events.csv"
    pipe.events_dataframe().to_csv(csv_path, index=False)
    return csv_path
