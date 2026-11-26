"""Tableau de bord temps réel (Streamlit).

Lancement : streamlit run app/dashboard.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from vtc.config import load_config
from vtc.pipeline import run_video

st.set_page_config(page_title="Comptage de véhicules", layout="wide")
st.title("🚗 Détection, suivi et comptage de véhicules en temps réel")
st.caption("YOLO + ByteTrack · ligne de comptage virtuelle · estimation de vitesse")

# --------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("Source")
    uploaded = st.file_uploader("Vidéo", type=["mp4", "avi", "mov", "mkv"])
    path_text = st.text_input("…ou chemin / URL / index de webcam", "")

    st.header("Modèle")
    model = st.selectbox("Poids YOLO", ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"])
    custom_weights = st.text_input("Poids personnalisés (best.pt)", "")
    conf = st.slider("Seuil de confiance", 0.05, 0.9, 0.30, 0.05)
    imgsz = st.select_slider("Résolution d'inférence", [480, 640, 800, 960, 1280], value=960)

    st.header("Ligne de comptage")
    line_y = st.slider("Position verticale (fraction de la hauteur)", 0.1, 0.95, 0.60, 0.01)
    flip = st.checkbox("Inverser les sens in / out", False)

    st.header("Vitesse (km/h)")
    speed_on = st.checkbox("Activer l'estimation de vitesse", False)
    width_m = st.number_input("Largeur réelle de la zone (m)", 1.0, 50.0, 7.0, 0.5)
    length_m = st.number_input("Longueur réelle de la zone (m)", 3.0, 200.0, 30.0, 1.0)
    st.caption("Zone par défaut modifiable dans le YAML (voir scripts/calibrate.py).")

    frame_skip = st.slider("Rafraîchir l'affichage toutes les N images", 1, 10, 2)
    start = st.button("▶ Lancer", type="primary", use_container_width=True)

# --------------------------------------------------------------------- mise en page
video_col, stats_col = st.columns([3, 2])
video_slot = video_col.empty()
with stats_col:
    m1, m2, m3, m4 = st.columns(4)
    kpi_total, kpi_in, kpi_out, kpi_fps = m1.empty(), m2.empty(), m3.empty(), m4.empty()
    kpi_speed = st.empty()
    st.subheader("Évolution du comptage cumulé")
    chart_slot = st.empty()
    st.subheader("Répartition par classe")
    bar_slot = st.empty()
st.subheader("Derniers franchissements")
table_slot = st.empty()
download_slot = st.empty()


def resolve_source() -> str | int | None:
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.getbuffer())
        return str(tmp)
    if path_text.strip():
        return int(path_text) if path_text.strip().isdigit() else path_text.strip()
    return None


if start:
    source = resolve_source()
    if source is None:
        st.warning("Choisissez une vidéo ou saisissez un chemin.")
        st.stop()

    cfg = load_config("configs/default.yaml")
    cfg.detector.model = custom_weights.strip() or model
    cfg.detector.conf, cfg.detector.imgsz = conf, imgsz
    cfg.counting.line = [[0.0, line_y], [1.0, line_y]]
    cfg.counting.labels = ["out", "in"] if flip else ["in", "out"]
    cfg.speed.enabled, cfg.speed.width_m, cfg.speed.length_m = speed_on, width_m, length_m

    history: list[dict] = []
    pipe = None
    try:
        with st.spinner("Chargement du modèle…"):
            gen = run_video(source, cfg)
        for pipe, res in gen:
            history.append({"time_s": res.timestamp_s, "total": sum(res.totals.values()), **res.totals})
            if res.frame_idx % frame_skip:
                continue

            video_slot.image(cv2.cvtColor(res.frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            l0, l1 = cfg.counting.labels
            kpi_total.metric("Total", sum(res.totals.values()))
            kpi_in.metric(l0.upper(), res.totals[l0])
            kpi_out.metric(l1.upper(), res.totals[l1])
            kpi_fps.metric("FPS", f"{res.fps_processing:.1f}")
            if speed_on:
                kpi_speed.metric(
                    "Vitesse moyenne (zone)",
                    "—" if res.mean_speed_kmh is None else f"{res.mean_speed_kmh:.0f} km/h",
                )

            df_hist = pd.DataFrame(history).set_index("time_s")[[l0, l1]]
            chart_slot.line_chart(df_hist)

            per_class = pd.DataFrame(res.counts).T.fillna(0).astype(int) if res.counts else pd.DataFrame()
            if not per_class.empty:
                bar_slot.bar_chart(per_class)

            ev_df = pipe.events_dataframe().tail(10).iloc[::-1]
            table_slot.dataframe(ev_df, use_container_width=True, hide_index=True)
        st.success("Traitement terminé.")
    except FileNotFoundError as e:
        st.error(str(e))
    finally:
        if pipe is not None:
            csv = pipe.events_dataframe().to_csv(index=False).encode("utf-8")
            download_slot.download_button("⬇ Télécharger les événements (CSV)", csv, "events.csv", "text/csv")
else:
    st.info("Configurez la source et les paramètres à gauche, puis cliquez sur **Lancer**.")
