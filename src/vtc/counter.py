"""Comptage de véhicules par franchissement d'une ligne virtuelle."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .geometry import Point, segments_intersect, side_of_line


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    direction: str  # "in" ou "out"
    frame_idx: int
    timestamp_s: float
    speed_kmh: float | None = None


@dataclass
class LineCounter:
    """Compte les franchissements d'une ligne (a, b), par classe et par sens.

    Principe : pour chaque identifiant de suivi, on mémorise la position précédente.
    Si le segment [position précédente -> position courante] coupe la ligne, on
    déclare un franchissement. Le sens est donné par le côté de départ.

    Parameters
    ----------
    a, b : extrémités de la ligne en pixels.
    labels : (label si passage du côté + vers -, label si passage de - vers +).
    count_once : si True, un même identifiant n'est compté qu'une fois
        (évite les doubles comptages si le véhicule oscille autour de la ligne).
    """

    a: Point
    b: Point
    labels: tuple[str, str] = ("in", "out")
    count_once: bool = True
    counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    _last_pos: dict[int, Point] = field(default_factory=dict)
    _counted: set[int] = field(default_factory=set)

    def update(
        self,
        track_id: int,
        position: Point,
        class_name: str,
        frame_idx: int,
        timestamp_s: float,
    ) -> CrossingEvent | None:
        prev = self._last_pos.get(track_id)
        self._last_pos[track_id] = position
        if prev is None:
            return None
        if self.count_once and track_id in self._counted:
            return None
        if not segments_intersect(prev, position, self.a, self.b):
            return None

        side_before = side_of_line(self.a, self.b, prev)
        if side_before == 0:  # on partait pile sur la ligne : ambigu, on attend l'image suivante
            return None
        direction = self.labels[0] if side_before > 0 else self.labels[1]

        self._counted.add(track_id)
        self.counts[class_name][direction] += 1
        return CrossingEvent(track_id, class_name, direction, frame_idx, timestamp_s)

    def forget(self, active_ids: set[int]) -> None:
        """Libère la mémoire des identifiants qui ont disparu (suivi long)."""
        for tid in list(self._last_pos):
            if tid not in active_ids and tid in self._counted:
                # on garde _counted pour ne pas recompter si l'ID réapparaît
                self._last_pos.pop(tid, None)

    @property
    def total(self) -> int:
        return sum(n for per_dir in self.counts.values() for n in per_dir.values())

    def totals_by_direction(self) -> dict[str, int]:
        out = {self.labels[0]: 0, self.labels[1]: 0}
        for per_dir in self.counts.values():
            for d, n in per_dir.items():
                out[d] += n
        return out

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {c: dict(d) for c, d in self.counts.items()}
