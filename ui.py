"""ui.py - Streamlit widgets only (sidebar, layout, metrics, tables)."""
import cv2
import pandas as pd
import streamlit as st


def render_sidebar(class_list):
    st.sidebar.title("⚙️ Controls")

    source = st.sidebar.radio("Video source", ["Webcam", "Video file"])
    camera_index = 0
    video_file = None
    if source == "Webcam":
        camera_index = st.sidebar.number_input("Camera index", 0, 10, 0)
    else:
        video_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    st.sidebar.divider()
    default = [c for c in ["person", "cell phone"] if c in class_list]
    selected = st.sidebar.multiselect(
        "Objects to detect (empty = all)", class_list, default=default)
    conf = st.sidebar.slider("Confidence threshold", 0.10, 0.99, 0.70, 0.05)

    st.sidebar.divider()
    log_to_db = st.sidebar.checkbox("Save detections to MySQL", value=True)
    interval = st.sidebar.slider("Log each class at most every (sec)", 0.0, 10.0, 2.0, 0.5)
    clear = st.sidebar.button("🗑️ Clear all logs")

    st.sidebar.divider()
    run = st.sidebar.toggle("▶️ Start detection", value=False)

    return {
        "source": source, "camera_index": int(camera_index), "video_file": video_file,
        "classes": selected, "conf": conf, "log_to_db": log_to_db,
        "interval": interval, "clear": clear, "run": run,
    }


def create_layout():
    st.title("🎯 Real-Time Object Detection Platform")
    st.caption("YOLOv8 + OpenCV + MySQL + Streamlit")
    video_col, info_col = st.columns([3, 2])
    with video_col:
        video_ph = st.empty()
    with info_col:
        metrics_ph = st.empty()
        st.subheader("🗄️ Latest database logs")
        logs_ph = st.empty()
    return video_ph, metrics_ph, logs_ph


def show_frame(placeholder, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    placeholder.image(rgb, channels="RGB", use_container_width=True)


def show_metrics(placeholder, fps, in_frame, session_total, db_total):
    with placeholder.container():
        c1, c2 = st.columns(2)
        c1.metric("FPS", f"{fps:.1f}")
        c2.metric("Objects in frame", in_frame)
        c3, c4 = st.columns(2)
        c3.metric("Detections (session)", session_total)
        c4.metric("Rows in MySQL", db_total)


def show_logs(placeholder, rows):
    if rows:
        placeholder.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        placeholder.info("No logs yet.")
