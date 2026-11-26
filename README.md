# 🚗 Détection, suivi et comptage de véhicules en temps réel

> Projet de Vision par Ordinateur Avancée — 3ᵉ année cycle ingénieur en informatique
> Domaine : **Mobilité / Smart City**

Pipeline complet qui, à partir d'une vidéo de trafic, **détecte** les véhicules (YOLOv8 / YOLO11), les **suit** d'image en image (ByteTrack), les **compte** lorsqu'ils franchissent une **ligne virtuelle** (par sens et par classe), **estime leur vitesse** (homographie image → plan de la route) et affiche le tout dans un **tableau de bord temps réel** (Streamlit).

```
vidéo ─► YOLO (détection) ─► ByteTrack (IDs) ─► LineCounter ─► compteurs in/out par classe
                                       └──────► SpeedEstimator (H) ─► km/h par véhicule
                                                        └─► annotation + CSV + dashboard Streamlit
```

## Ce qui vient de nous (contributions) vs. ressources externes

| Développé par nous | Repris de l'extérieur |
|---|---|
| `src/vtc/` : comptage par franchissement de ligne (`counter.py`, `geometry.py`), estimation de vitesse par homographie (`speed.py`), pipeline et annotation (`pipeline.py`), configuration (`config.py`) | Modèles YOLOv8 / YOLO11 et tracker ByteTrack via la bibliothèque [Ultralytics](https://github.com/ultralytics/ultralytics) (licence AGPL-3.0) |
| `app/dashboard.py` : tableau de bord temps réel | Jeux de données VisDrone / UA-DETRAC (voir licences ci-dessous) |
| `scripts/` : calibration interactive, conversion VisDrone→YOLO, entraînement, évaluation du comptage, benchmark | Algorithme ByteTrack : Zhang et al., ECCV 2022 |
| `tests/` : tests unitaires (géométrie, comptage, vitesse, pipeline avec détecteur factice) | |

## Installation

```bash
git clone https://github.com/<votre-compte>/vehicle-tracking-counting.git
cd vehicle-tracking-counting
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
# (GPU NVIDIA) installez d'abord PyTorch adapté à votre CUDA : https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -e .
pytest -q                                              # 9 tests, ne nécessitent pas de GPU
```

## Utilisation

### 1. Lancer sur une vidéo

```bash
python scripts/run_video.py --source data/videos/traffic.mp4 --config configs/default.yaml --show
```
Sorties dans `outputs/` : vidéo annotée, `*_events.csv` (un franchissement par ligne : id, classe, sens, instant, vitesse).

### 2. Calibrer la ligne et la zone de vitesse (une fois par caméra)

```bash
python scripts/calibrate.py --source data/videos/traffic.mp4
```
Cliquez 2 points (ligne), puis 4 points au sol (zone de vitesse). Copiez le YAML affiché dans votre config et renseignez `width_m` / `length_m` avec les dimensions **réelles** de la zone.

### 3. Tableau de bord temps réel

```bash
streamlit run app/dashboard.py
```

### 4. (Option) Fine-tuning sur VisDrone

```bash
python scripts/convert_visdrone_to_yolo.py --root data/VisDrone \
    --splits VisDrone2019-DET-train VisDrone2019-DET-val --vehicles-only
python scripts/train.py --data data/visdrone_yolo/data.yaml --model yolov8s.pt --epochs 50 --imgsz 960 --batch 8
python scripts/evaluate.py --model runs/train/exp/weights/best.pt --data data/visdrone_yolo/data.yaml
```

### 5. Mesurer la qualité

```bash
python scripts/eval_counting.py --events outputs/traffic_events.csv --gt-in 87 --gt-out 74   # vs comptage manuel
python scripts/benchmark.py --source data/videos/traffic.mp4 --models yolov8n.pt yolov8s.pt --imgsz 640 960
```

## Résultats

> ⚠️ **À compléter avec VOS mesures** après exécution (ne laissez aucun chiffre non mesuré).

| Modèle | imgsz | mAP50 | mAP50-95 | FPS (GPU/CPU) | Erreur de comptage |
|---|---|---|---|---|---|
| YOLOv8n (COCO) | 640 | … | … | … | … % |
| YOLOv8s (fine-tuné) | 960 | … | … | … | … % |

Démo : `assets/demo.mp4` (à ajouter) · GIF : `assets/demo.gif`

## Reproductibilité

- **Code** : ce dépôt, tag `v1.0.0` (commit : `<hash>`).
- **Données** : [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) (usage académique / non commercial — vérifier la licence sur la page officielle) ; [UA-DETRAC](https://detrac-db.rit.albany.edu/) (idem, vérifier les conditions) ; versions au format YOLO sur Kaggle / Roboflow (vérifier la licence de chaque version).
- **Environnement** : voir `docs/ENVIRONMENT.md` (à remplir avec `pip freeze`).
- Graines aléatoires : `--seed` dans `train.py`.

## Limites connues

- Vitesse : hypothèse d'un **sol plan** et d'une calibration correcte ; précision dégradée loin de la caméra.
- Occlusions fortes / embouteillages : changements d'identité (ID switches) possibles → risque de sous- ou sur-comptage.
- Vues de drone (VisDrone) : objets très petits ; le comptage par ligne est conçu pour une caméra fixe.
- Pas de plaque / réidentification : on ne suit que dans un seul champ de vue.

## Références

- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
- Jocher et al., Ultralytics YOLO — <https://docs.ultralytics.com>
- Zhu et al., *Detection and Tracking Meet Drones Challenge* (VisDrone), IEEE TPAMI 2021.
- Wen et al., *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*, CVIU 2020.

## Licence

Code sous licence MIT (voir `LICENSE`). Attention : Ultralytics est distribué sous AGPL-3.0, ce qui s'applique si vous redistribuez un service basé sur ces poids/bibliothèque.
