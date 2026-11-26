"""Chargement de la configuration YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DetectorCfg:
    model: str = "yolov8n.pt"
    tracker: str = "bytetrack.yaml"
    conf: float = 0.30
    iou: float = 0.50
    imgsz: int = 960
    device: str | None = None
    vehicle_classes: list[str] = field(
        default_factory=lambda: ["car", "motorcycle", "bus", "truck"]
    )


@dataclass
class CountingCfg:
    # Coordonnées normalisées (0-1) -> indépendantes de la résolution de la vidéo
    line: list[list[float]] = field(default_factory=lambda: [[0.0, 0.6], [1.0, 0.6]])
    labels: list[str] = field(default_factory=lambda: ["in", "out"])
    count_once: bool = True
    anchor: str = "bottom_center"  # "bottom_center" ou "center"


@dataclass
class SpeedCfg:
    enabled: bool = False
    # 4 points normalisés : haut-gauche, haut-droit, bas-droit, bas-gauche
    src_points: list[list[float]] = field(
        default_factory=lambda: [[0.30, 0.45], [0.70, 0.45], [0.90, 0.90], [0.10, 0.90]]
    )
    width_m: float = 7.0  # largeur réelle de la zone (ex. 2 voies)
    length_m: float = 30.0  # longueur réelle de la zone
    window_s: float = 1.0
    max_kmh: float = 200.0


@dataclass
class OutputCfg:
    out_dir: str = "outputs"
    save_video: bool = True
    save_events_csv: bool = True
    trail_length: int = 30


@dataclass
class Config:
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    counting: CountingCfg = field(default_factory=CountingCfg)
    speed: SpeedCfg = field(default_factory=SpeedCfg)
    output: OutputCfg = field(default_factory=OutputCfg)


def _merge(dc, data: dict):
    for k, v in (data or {}).items():
        if not hasattr(dc, k):
            raise KeyError(f"Clé de configuration inconnue : {k!r}")
        setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section, dc in (
        ("detector", cfg.detector),
        ("counting", cfg.counting),
        ("speed", cfg.speed),
        ("output", cfg.output),
    ):
        _merge(dc, data.get(section, {}))
    return cfg
