"""main.py - entry point. Run with:  streamlit run main.py"""
import os
import tempfile
import time

import cv2
import streamlit as st

import config
import ui
from database import DatabaseManager
from detector import ObjectDetector

st.set_page_config(page_title="Object Detection Platform", page_icon="🎯", layout="wide")


@st.cache_resource
def load_detector(path):
    return ObjectDetector(path)


@st.cache_resource
def load_database():
    db = DatabaseManager(config.DB_CONFIG)
    db.initialize()
    return db


def open_source(settings):
    if settings["source"] == "Webcam":
        return cv2.VideoCapture(settings["camera_index"])
    f = settings["video_file"]
    if f is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1])
    tmp.write(f.read())
    tmp.close()
    return cv2.VideoCapture(tmp.name)


def main():
    detector = load_detector(config.MODEL_PATH)

    try:
        db = load_database()
    except Exception as e:
        db = None
        st.sidebar.error(f"MySQL not connected: {e}")

    settings = ui.render_sidebar(detector.get_class_list())
    video_ph, metrics_ph, logs_ph = ui.create_layout()

    if settings["clear"] and db:
        db.clear_logs()

    if db:
        ui.show_logs(logs_ph, db.get_recent_logs(10))
    ui.show_metrics(metrics_ph, 0, 0, 0, db.count_logs() if db else 0)

    if not settings["run"]:
        video_ph.info("Turn on **Start detection** in the sidebar.")
        return

    cap = open_source(settings)
    if cap is None or not cap.isOpened():
        video_ph.error("Could not open the video source (check camera index / upload a file).")
        return

    last_logged = {}
    session_total = 0
    last_log_refresh = 0
    prev = time.time()

    while settings["run"] and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            st.info("Video finished.")
            break

        annotated, dets = detector.detect(frame, settings["conf"], settings["classes"])
        session_total += len(dets)

        # Throttle logging so the DB isn't flooded (once per class per interval)
        now = time.time()
        if db and settings["log_to_db"] and dets:
            due = {c for c in {d["object_class"] for d in dets}
                   if now - last_logged.get(c, 0) >= settings["interval"]}
            to_log = [d for d in dets if d["object_class"] in due]
            if to_log:
                db.insert_detections(to_log)
                for c in due:
                    last_logged[c] = now

        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        ui.show_frame(video_ph, annotated)

        if now - last_log_refresh >= 1.0:  # refresh table + metrics once per second
            rows = db.get_recent_logs(10) if db else []
            total = db.count_logs() if db else 0
            ui.show_logs(logs_ph, rows)
            ui.show_metrics(metrics_ph, fps, len(dets), session_total, total)
            last_log_refresh = now

    cap.release()


if __name__ == "__main__":
    main()
