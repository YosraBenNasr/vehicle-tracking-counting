"""Primitives géométriques utilisées par le compteur de franchissement de ligne."""

from __future__ import annotations

Point = tuple[float, float]


def cross(a: Point, b: Point, c: Point) -> float:
    """Produit vectoriel (b - a) x (c - a).

    > 0 : c est à gauche du vecteur a->b, < 0 : à droite, 0 : aligné.
    (Dans un repère image où y pointe vers le bas, "gauche/droite" sont inversés
    visuellement, mais seul le *signe* compte pour détecter un changement de côté.)
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def side_of_line(a: Point, b: Point, p: Point) -> int:
    """Renvoie +1, -1 ou 0 selon le côté de p par rapport à la droite (a, b)."""
    v = cross(a, b, p)
    return (v > 0) - (v < 0)


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Vrai si le segment [p1, p2] coupe le segment [q1, q2] (test d'orientation).

    p1->p2 : déplacement d'un objet entre deux images.
    q1->q2 : ligne de comptage (de longueur finie).
    """
    d1 = side_of_line(q1, q2, p1)
    d2 = side_of_line(q1, q2, p2)
    d3 = side_of_line(p1, p2, q1)
    d4 = side_of_line(p1, p2, q2)

    # Cas général : les extrémités de chaque segment sont de part et d'autre de l'autre
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # Cas dégénérés (points colinéaires) : on vérifie l'appartenance à la boîte englobante
    def on_segment(a: Point, b: Point, p: Point) -> bool:
        return (
            min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        )

    if d1 == 0 and on_segment(q1, q2, p1):
        return True
    if d2 == 0 and on_segment(q1, q2, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, q1):
        return True
    if d4 == 0 and on_segment(p1, p2, q2):
        return True
    return False
