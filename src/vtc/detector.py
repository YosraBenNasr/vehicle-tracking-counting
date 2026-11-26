"""Détection + suivi : enveloppe autour d'Ultralytics YOLO et du tracker ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    cls_id: int
    cls_name: str
    conf: float


class TrackerBackend(Protocol):
    """Toute implémentation (YOLO réel, faux détecteur de test...) doit offrir ceci."""

    def track(self, frame: np.ndarray) -> list[Track]: ...


class YoloByteTrack:
    """YOLOv8 / YOLO11 + ByteTrack via ``model.track(..., persist=True)``.

    ByteTrack associe *toutes* les détections, y compris celles à faible score
    (seconde association), ce qui limite les pertes d'identité lors des occlusions.
    Les hyperparamètres du tracker (track_high_thresh, track_buffer...) se règlent
    dans un YAML passé via ``tracker=``.
    """

    def __init__(
        self,
        model: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.30,
        iou: float = 0.50,
        imgsz: int = 960,
        device: str | None = None,
        vehicle_classes: list[str] | None = None,
    ) -> None:
        from ultralytics import YOLO  # import tardif : le reste du paquet reste testable sans torch

        self.model = YOLO(model)
        self.tracker = tracker
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        names: dict[int, str] = self.model.names
        wanted = set(vehicle_classes or names.values())
        self.class_ids = [i for i, n in names.items() if n in wanted]
        if not self.class_ids:
            raise ValueError(
                f"Aucune classe de {sorted(wanted)} dans le modèle (classes : {list(names.values())})"
            )
        self.names = names

    def track(self, frame: np.ndarray) -> list[Track]:
        res = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:  # aucun objet suivi dans cette image
            return []
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.int().cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        return [
            Track(i, tuple(map(float, b)), c, self.names[c], float(s))
            for i, b, c, s in zip(ids, xyxy, cls, conf)
        ]
