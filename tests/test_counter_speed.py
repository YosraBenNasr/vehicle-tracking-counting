"""Tests unitaires : géométrie, comptage, vitesse, et pipeline avec faux détecteur."""

import numpy as np
import pytest

from vtc.config import load_config
from vtc.counter import LineCounter
from vtc.detector import Track
from vtc.geometry import segments_intersect, side_of_line
from vtc.pipeline import build_pipeline
from vtc.speed import SpeedEstimator


# ------------------------------------------------------------------ géométrie
def test_side_of_line_signs():
    assert side_of_line((0, 0), (10, 0), (5, 3)) == -side_of_line((0, 0), (10, 0), (5, -3))
    assert side_of_line((0, 0), (10, 0), (5, 0)) == 0


def test_segments_intersect_basic():
    assert segments_intersect((5, -1), (5, 1), (0, 0), (10, 0))
    assert not segments_intersect((5, 1), (5, 3), (0, 0), (10, 0))
    # croise la droite mais hors du segment fini
    assert not segments_intersect((20, -1), (20, 1), (0, 0), (10, 0))


# ------------------------------------------------------------------- comptage
def test_counts_each_vehicle_once_and_by_direction():
    c = LineCounter((0, 100), (200, 100), labels=("in", "out"))
    # véhicule 1 : monte (y décroît), véhicule 2 : descend
    assert c.update(1, (50, 120), "car", 0, 0.0) is None
    ev1 = c.update(1, (50, 80), "car", 1, 0.04)
    assert ev1 is not None
    assert c.update(2, (150, 80), "truck", 0, 0.0) is None
    ev2 = c.update(2, (150, 120), "truck", 1, 0.04)
    assert ev2 is not None
    assert ev1.direction != ev2.direction
    # oscillation autour de la ligne : pas de double comptage
    assert c.update(1, (50, 120), "car", 2, 0.08) is None
    assert c.update(1, (50, 80), "car", 3, 0.12) is None
    assert c.total == 2


def test_no_count_when_not_crossing():
    c = LineCounter((0, 100), (200, 100))
    for i, y in enumerate([10, 20, 30, 40, 90]):
        assert c.update(7, (100, y), "car", i, i / 25) is None
    assert c.total == 0


# -------------------------------------------------------------------- vitesse
def test_speed_estimation_constant_motion():
    # Zone image 400x400 px <-> 10 m x 40 m (vue du dessus, pas de perspective)
    src = [(0, 0), (400, 0), (400, 400), (0, 400)]
    se = SpeedEstimator(src, width_m=10, length_m=40, fps=25, window_s=1.0)
    # 100 px/s en y = 10 m/s = 36 km/h (400 px <-> 40 m => 0.1 m/px)
    v = None
    for f in range(0, 60):
        y = 50 + f * (100 / 25)  # 4 px/frame
        v = se.update(1, (200, y), f)
    assert v is not None
    assert v == pytest.approx(36.0, rel=0.05)


def test_speed_outside_zone_returns_none():
    src = [(100, 100), (300, 100), (300, 300), (100, 300)]
    se = SpeedEstimator(src, 5, 20, fps=25)
    assert se.update(1, (10, 10), 0) is None


def test_speed_requires_four_points():
    with pytest.raises(ValueError):
        SpeedEstimator([(0, 0), (1, 1)], 5, 5, fps=25)


# ----------------------------------------------- pipeline avec détecteur factice
class FakeDetector:
    """Simule deux véhicules qui traversent la ligne dans des sens opposés."""

    def __init__(self):
        self.n = 0

    def track(self, frame):
        i, self.n = self.n, self.n + 1
        # image 200x200, ligne à y = 0.6*200 = 120
        y_down = 40 + 4 * i  # descend
        y_up = 180 - 4 * i  # monte
        return [
            Track(1, (80, y_down - 20, 120, y_down), 2, "car", 0.9),
            Track(2, (130, y_up - 20, 170, y_up), 7, "truck", 0.9),
        ]


def test_pipeline_end_to_end_with_fake_detector():
    cfg = load_config()
    cfg.speed.enabled = False
    pipe = build_pipeline(cfg, fps=25, size=(200, 200), detector=FakeDetector())
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(40):
        res = pipe.process_frame(frame, i)
        assert res.frame.shape == frame.shape
    assert pipe.counter.total == 2
    df = pipe.events_dataframe()
    assert set(df.class_name) == {"car", "truck"}
    assert set(df.direction) == {"in", "out"}


def test_load_config_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("detector:\n  modle: x\n")
    with pytest.raises(KeyError):
        load_config(p)
