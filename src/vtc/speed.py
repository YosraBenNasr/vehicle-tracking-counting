"""Estimation de vitesse par homographie (image -> plan de la route en mètres)."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .geometry import Point


class SpeedEstimator:
    """Convertit les positions image en positions métriques puis en km/h.

    Calibration : 4 points de l'image (coin d'une zone rectangulaire au sol, ordre
    haut-gauche, haut-droit, bas-droit, bas-gauche) dont on connaît les dimensions
    réelles ``width_m`` x ``length_m``. On calcule l'homographie H qui projette
    l'image sur le plan de la route vu du dessus.

    La vitesse est calculée sur une fenêtre glissante (``window_s`` secondes), puis
    lissée par moyenne mobile exponentielle pour limiter le bruit de détection.

    Limites (à mentionner à l'oral) : hypothèse d'un sol plan, précision dépendante
    de la qualité de la calibration, point de contact approximé par le bas de la boîte.
    """

    def __init__(
        self,
        src_points_px: list[Point],
        width_m: float,
        length_m: float,
        fps: float,
        window_s: float = 1.0,
        min_window_s: float = 0.4,
        max_kmh: float = 200.0,
        ema_alpha: float = 0.3,
    ) -> None:
        if len(src_points_px) != 4:
            raise ValueError("Il faut exactement 4 points de calibration.")
        self.fps = float(fps)
        self.window_frames = max(2, int(round(window_s * fps)))
        self.min_frames = max(2, int(round(min_window_s * fps)))
        self.max_kmh = max_kmh
        self.alpha = ema_alpha

        src = np.asarray(src_points_px, dtype=np.float32)
        dst = np.asarray(
            [[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]], dtype=np.float32
        )
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.polygon = src.reshape(-1, 1, 2)

        self._hist: dict[int, deque[tuple[int, float, float]]] = {}
        self._speed: dict[int, float] = {}

    # -- utilitaires ---------------------------------------------------------
    def to_world(self, p: Point) -> tuple[float, float]:
        pt = np.array([[[p[0], p[1]]]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(pt, self.H)[0, 0]
        return float(x), float(y)

    def in_zone(self, p: Point) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(p[0]), float(p[1])), False) >= 0

    # -- API principale ------------------------------------------------------
    def update(self, track_id: int, position: Point, frame_idx: int) -> float | None:
        """Met à jour l'historique du véhicule et renvoie sa vitesse lissée (km/h)."""
        if not self.in_zone(position):
            return self._speed.get(track_id)  # hors zone calibrée : on garde la dernière valeur

        x, y = self.to_world(position)
        hist = self._hist.setdefault(track_id, deque(maxlen=self.window_frames))
        hist.append((frame_idx, x, y))
        if len(hist) < 2:
            return self._speed.get(track_id)

        f0, x0, y0 = hist[0]
        f1, x1, y1 = hist[-1]
        if f1 - f0 < self.min_frames:
            return self._speed.get(track_id)

        dt = (f1 - f0) / self.fps
        dist_m = float(np.hypot(x1 - x0, y1 - y0))
        kmh = dist_m / dt * 3.6
        if kmh > self.max_kmh:  # valeur aberrante (saut d'ID, mauvaise détection)
            return self._speed.get(track_id)

        prev = self._speed.get(track_id)
        self._speed[track_id] = kmh if prev is None else self.alpha * kmh + (1 - self.alpha) * prev
        return self._speed[track_id]

    def get(self, track_id: int) -> float | None:
        return self._speed.get(track_id)

    def drop(self, active_ids: set[int]) -> None:
        for tid in list(self._hist):
            if tid not in active_ids:
                self._hist.pop(tid, None)
