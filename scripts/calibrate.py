#!/usr/bin/env python
"""Outil interactif de calibration : ligne de comptage + zone de vitesse.

Usage : python scripts/calibrate.py --source data/videos/traffic.mp4

1. Cliquez 2 points  -> ligne de comptage
2. Cliquez 4 points  -> zone au sol (haut-gauche, haut-droit, bas-droit, bas-gauche)
3. Appuyez sur Entrée : le bloc YAML (coordonnées normalisées) est affiché.
   Renseignez ensuite width_m / length_m (dimensions réelles de la zone, p. ex. une
   portion de route dont vous connaissez la largeur et la longueur : marquages au sol,
   longueur d'une rangée de pointillés = 3 m en France, etc.).
"""

from __future__ import annotations

import argparse

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=0, help="numéro de l'image à utiliser")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Lecture de l'image impossible")
    h, w = frame.shape[:2]

    pts: list[tuple[int, int]] = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 6:
            pts.append((x, y))

    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", on_click)
    while True:
        view = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 5, (0, 0, 255) if i < 2 else (0, 200, 255), -1)
        if len(pts) >= 2:
            cv2.line(view, pts[0], pts[1], (0, 0, 255), 2)
        if len(pts) > 2:
            for a, b in zip(pts[2:], pts[3:]):
                cv2.line(view, a, b, (0, 200, 255), 2)
            if len(pts) == 6:
                cv2.line(view, pts[5], pts[2], (0, 200, 255), 2)
        cv2.imshow("calibration", view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and len(pts) in (2, 6):
            break
        if k == ord("q"):
            return
    cv2.destroyAllWindows()

    n = lambda p: [round(p[0] / w, 4), round(p[1] / h, 4)]  # noqa: E731
    print("\n# --- à copier dans votre fichier YAML ---")
    print("counting:")
    print(f"  line: [{n(pts[0])}, {n(pts[1])}]")
    if len(pts) == 6:
        print("speed:")
        print("  enabled: true")
        print(f"  src_points: [{n(pts[2])}, {n(pts[3])}, {n(pts[4])}, {n(pts[5])}]")
        print("  width_m: ???    # largeur réelle de la zone, en mètres")
        print("  length_m: ???   # longueur réelle de la zone, en mètres")


if __name__ == "__main__":
    main()
