import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os

# Create folder for saved images
SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection")

# Sidebar settings
target_object = st.sidebar.selectbox(
    "Alert Object:",
    ["person", "cell phone", "bottle"]
)

save_frames = st.sidebar.checkbox("Save Frames")

last_save = 0


def video_frame_callback(frame):
    global last_save

    img = frame.to_ndarray(format="bgr24")

    # Detect objects
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    result = results[0]

    if result.boxes:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = model.names
        labels = [names[i] for i in class_ids]

        # Count persons
        person_count = labels.count("person")

        # Show count
        cv2.putText(img, f"People: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Alert message
        if target_object in labels:
            cv2.putText(img, f"{target_object} detected!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save frame
        if save_frames and len(labels) > 0:
            now = time.time()
            if now - last_save > 3:
                cv2.imwrite(f"{SAVE_DIR}/frame_{int(now)}.jpg", img)
                last_save = now

    return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")


webrtc_streamer(
    key="detect",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
)