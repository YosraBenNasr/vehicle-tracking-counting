# Journal des difficultés techniques

> ⚠️ **Ce fichier est un gabarit.** La consigne du cours exige des difficultés **réellement rencontrées**
> (capture d'écran / log à l'appui). Remplacez chaque entrée ci-dessous par ce que VOUS observez en exécutant
> le projet ; supprimez celles qui ne vous sont pas arrivées. Les causes/solutions listées sont les pièges les
> plus fréquents avec ce type de projet, pour vous aider à diagnostiquer.

## Modèle de fiche

1. **Problème rencontré** :
2. **Message d'erreur / comportement observé** (log copié tel quel, capture) :
3. **Causes possibles** (hypothèses testées) :
4. **Solutions envisagées / retenue** :
5. **Résultat après correction** (preuve : log, capture, courbe avant/après) :

## Pistes fréquentes (à vérifier chez vous)

### A. `CUDA out of memory` à l'entraînement
- Causes : `imgsz` trop grand (VisDrone : petits objets → on est tenté de monter à 1280), `batch` trop élevé.
- Solutions : `--batch 4`, `--imgsz 960`, modèle `n`/`s`, précision mixte (activée par défaut), `workers` réduit.
- Preuve : log avant (erreur) / log après (epoch terminé), `nvidia-smi`.

### B. ID qui changent (ID switches) → sur-comptage
- Observation : un même camion reçoit plusieurs identifiants quand il est occlus ; le compteur dépasse la vérité terrain.
- Causes : détections instables, `track_buffer` trop court, seuil `conf` trop haut (la seconde association de ByteTrack a besoin de détections faibles).
- Solutions : baisser `conf` à 0,2–0,3, augmenter `track_buffer` (`configs/bytetrack_custom.yaml`), fine-tuner, `count_once: true`.
- Preuve : erreur de comptage avant/après avec `scripts/eval_counting.py`.

### C. Vitesses absurdes (ex. 400 km/h ou 3 km/h)
- Causes : ordre des 4 points de calibration incorrect, `width_m` / `length_m` inversés, zone trop petite (bruit), ID qui saute.
- Solutions : vérifier l'ordre (haut-gauche, haut-droit, bas-droit, bas-gauche), utiliser `calibrate.py`, fenêtre `window_s` = 1 s, filtre `max_kmh`.
- Preuve : comparer à une vitesse connue (véhicule à vitesse stabilisée, limite de la route).

### D. Couleurs / formats
- Streamlit attend du RGB, OpenCV produit du BGR → couleurs inversées : `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` (déjà fait dans `dashboard.py`).
- Annotations VisDrone : `x,y,w,h` en pixels (coin haut-gauche) ≠ YOLO normalisé centré → voir `convert_visdrone_to_yolo.py`.

### E. Incompatibilité PyTorch / CUDA / Python
- Symptôme : `torch.cuda.is_available()` renvoie `False` ou erreur de version.
- Solution : installer PyTorch via la commande officielle correspondant à votre version de CUDA, dans un environnement virtuel propre.
