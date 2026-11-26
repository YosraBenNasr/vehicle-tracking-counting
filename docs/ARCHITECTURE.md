# Architecture et choix techniques

## Flux de données (tenseurs)

```
Image BGR (H×W×3, uint8)
  └─ letterbox + normalisation ─► (1, 3, 960, 960) float32 [0,1]  (RGB)
      └─ Backbone + Neck (CSP / PAN-FPN) ─► cartes à 3 échelles : 120×120, 60×60, 30×30
          └─ Tête découplée, sans ancres ─► (1, 4+nc, N) avec N = 120²+60²+30² = 18 900
              └─ NMS (conf, IoU) ─► détections [x1, y1, x2, y2, score, classe]
                  └─ ByteTrack ─► [x1, y1, x2, y2, track_id, score, classe]
                      ├─ point d'ancrage (bas-centre de la boîte)
                      │    ├─ LineCounter : segment(prev→cur) ∩ ligne ? ─► événement (id, classe, sens)
                      │    └─ SpeedEstimator : H·(u,v,1) ─► (X,Y) en mètres ─► km/h
                      └─ annotation + CSV + dashboard
```

(`nc` = nombre de classes ; dimensions données pour une entrée 960×960.)

## Fonction de perte (entraînement YOLOv8/11)

`L = λ_box · L_CIoU + λ_cls · L_BCE + λ_dfl · L_DFL`, avec affectation dynamique des cibles (Task-Aligned Assigner).
Le comptage et la vitesse n'ont pas de perte : ce sont des modules géométriques déterministes.

## ByteTrack en bref

1. Prédiction de la position de chaque trajectoire par filtre de Kalman.
2. **Première association** (algorithme hongrois, distance IoU) entre trajectoires et détections de **score élevé**.
3. **Seconde association** des trajectoires restantes avec les détections de **score faible** (souvent des véhicules partiellement occlus que les autres trackers jettent).
4. Création de nouvelles trajectoires seulement à partir de détections de score élevé non associées ; une trajectoire perdue est conservée `track_buffer` images.

## Comptage par franchissement

Pour chaque ID, on teste si le segment [position précédente → position courante] coupe la ligne finie. Le **sens** vient du signe du produit vectoriel avant le franchissement. Un ID n'est compté qu'une fois (`count_once`) pour éviter les doubles comptages dus aux oscillations de la boîte.

## Vitesse par homographie

4 points image ↔ 4 points au sol (rectangle `width_m × length_m`) ⇒ `H = getPerspectiveTransform`. Distance parcourue en mètres sur une fenêtre glissante de 1 s ÷ durée, puis lissage exponentiel (α = 0,3) ; valeurs > `max_kmh` rejetées.

## Pourquoi ces choix ?

| Choix | Alternatives | Raison |
|---|---|---|
| YOLOv8/11 | Faster R-CNN, RT-DETR, SSD | Bon compromis précision/vitesse, tête sans ancres, écosystème (entraînement, export, tracker intégré) |
| ByteTrack | SORT, DeepSORT, BoT-SORT, OC-SORT | Pas de modèle d'apparence (rapide, simple), robuste aux occlusions grâce aux détections faibles ; DeepSORT ajoute un réseau de ré-identification coûteux |
| Franchissement de segment | Simple comparaison `y > y_ligne` | Robuste aux lignes obliques et aux grandes vitesses (un véhicule peut « sauter » la ligne entre deux images) |
| Homographie | Échelle pixel→mètre constante | Prend en compte la perspective (un pixel proche ≠ un pixel lointain) |
| Streamlit | Flask + JS, Gradio | Développement rapide en Python pur, widgets temps réel |
