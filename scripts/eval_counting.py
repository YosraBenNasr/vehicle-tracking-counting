#!/usr/bin/env python
"""Compare le comptage automatique à un comptage manuel (vérité terrain).

1. Regardez la vidéo (ou un extrait de 2-3 min) et comptez à la main les véhicules
   qui franchissent la ligne, par sens (et idéalement par classe).
2. Lancez : python scripts/eval_counting.py --events outputs/traffic_events.csv \
              --gt-in 87 --gt-out 74

Sortie : erreur absolue et erreur relative (%) par sens et au total.
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--gt-in", type=int, required=True)
    ap.add_argument("--gt-out", type=int, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.events)
    pred = {"in": int((df.direction == "in").sum()), "out": int((df.direction == "out").sum())}
    gt = {"in": args.gt_in, "out": args.gt_out}

    print(f"{'sens':<8}{'prédit':>8}{'réel':>8}{'err. abs':>10}{'err. rel.':>11}")
    tot_p = tot_g = 0
    for k in ("in", "out"):
        err = abs(pred[k] - gt[k])
        print(f"{k:<8}{pred[k]:>8}{gt[k]:>8}{err:>10}{err / max(gt[k], 1) * 100:>10.1f}%")
        tot_p += pred[k]
        tot_g += gt[k]
    err = abs(tot_p - tot_g)
    print(f"{'total':<8}{tot_p:>8}{tot_g:>8}{err:>10}{err / max(tot_g, 1) * 100:>10.1f}%")
    print(f"\nPrécision de comptage ≈ {100 - err / max(tot_g, 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
