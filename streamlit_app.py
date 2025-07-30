import streamlit as st
import cv2
import tempfile
import os
import time
from video_stream_tracking_appmodule import process_frame, fps

# --- Streamlit Page Config ---
st.set_page_config(page_title="Pedestrian Safety Assistant", layout="wide")
st.title("🚦 Pedestrian Safety Assistant")
st.markdown("Upload a video or use your live camera feed to analyze crossing safety.")

# --- Mode Selection ---
mode = st.radio("Choose input source:", ["Upload Video", "Live Camera"])

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        # Save uploaded video to a temp file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video.")
        else:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                fps = int(video_fps)

            stframe = st.empty()
            frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                processed_frame = process_frame(frame)

                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_container_width=True)

                time.sleep(1 / fps)

            cap.release()
            os.unlink(temp_video.name)
            st.success("✅ Video processing completed.")
    else:
        st.info("👆 Upload a video file to start analysis.")

elif mode == "Live Camera":
    stframe = st.empty()
    run = st.checkbox("Start Camera")

    if run:
        cap = cv2.VideoCapture(0)  # 0 = default webcam

        if not cap.isOpened():
            st.error("❌ Could not access your camera.")
        else:
            while run and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame(frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_container_width=True)

                time.sleep(1 / fps)

            cap.release()
            st.success("✅ Camera analysis stopped.")
