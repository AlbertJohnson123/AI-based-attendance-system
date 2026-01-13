import streamlit as st
import pandas as pd
import os
import cv2
import start_attendance
import stop_attendance
import face_recognition

# -------------------- BASE PATH (CRITICAL) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_faces")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Face Attendance Dashboard",
    page_icon="🧑‍💼",
    layout="wide"
)

# -------------------- HEADER --------------------
st.title("🧑‍💼 FACE ATTENDANCE SYSTEM")
st.caption("Real-time Recognition • Anti-Spoofing • Unknown Face Logging")
st.divider()

# -------------------- START / STOP BUTTONS --------------------
col1, col2 = st.columns(2)

if "attendance_status" not in st.session_state:
    st.session_state.attendance_status = "Stopped"

with col1:
    if st.button("▶ START ATTENDANCE", type="primary"):
        msg = start_attendance.start_attendance()
        st.session_state.attendance_status = "Running"
        st.success(msg)

with col2:
    if st.button("⏹ STOP ATTENDANCE"):
        msg = stop_attendance.stop_attendance()
        st.session_state.attendance_status = "Stopped"
        st.warning(msg)

st.info(f"📌 Attendance Status: {st.session_state.attendance_status}")
st.divider()

# -------------------- LIVE CAMERA SNAPSHOT (SAFE PREVIEW) --------------------
st.subheader("📸 Live Camera Snapshot (Preview Only)")

enable_cam = st.checkbox("Enable Camera Preview")

frame_placeholder = st.empty()

if enable_cam:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Webcam not available")
    else:
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb)

            for (t, r, b, l) in faces:
                cv2.rectangle(rgb, (l, t), (r, b), (0, 255, 0), 2)

            frame_placeholder.image(rgb, channels="RGB")
        cap.release()

st.divider()

# -------------------- ATTENDANCE TABLE --------------------
st.subheader("📒 Attendance Records")

if os.path.exists(ATTENDANCE_FILE):
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df, use_container_width=True, height=350)
else:
    st.warning("⚠ attendance.csv not found")

st.divider()

# -------------------- UNKNOWN FACE IMAGES --------------------
st.subheader("🖼 Unknown Face Images")

# Auto-create folder if missing (important fix)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

images = sorted(
    [f for f in os.listdir(UNKNOWN_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))],
    reverse=True
)

st.caption(f"Total Unknown Faces Captured: {len(images)}")

if images:
    cols = st.columns(4)
    for i, img in enumerate(images):
        with cols[i % 4]:
            st.image(
                os.path.join(UNKNOWN_DIR, img),
                caption=img,
                use_column_width=True
            )
else:
    st.info("No unknown faces captured yet.")

st.divider()

# -------------------- FOOTER --------------------
st.caption("© Face Attendance System | AI-powered Monitoring")
