import cv2
import face_recognition
import pickle
from datetime import datetime
import os
import csv
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance
import urllib.request
import bz2

# ==================== BASE PATH (VERY IMPORTANT) ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_faces")
UNKNOWN_VIDEO_DIR = os.path.join(BASE_DIR, "unknown_videos")
UNKNOWN_FULL_DIR = os.path.join(BASE_DIR, "unknown_full_videos")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# ==================== SETTINGS ====================
UNKNOWN_TIMEOUT = 30
FPS = 15
FACE_SIZE = (224, 224)

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
EAR_CONSEC_FRAMES_FOR_SPOOF = 50

# ==================== HELPERS ====================
def ensure_dirs():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_VIDEO_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_FULL_DIR, exist_ok=True)

def download_predictor():
    if not os.path.exists(PREDICTOR_PATH):
        print("[INFO] Downloading face landmark model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, PREDICTOR_PATH + ".bz2")
        with bz2.open(PREDICTOR_PATH + ".bz2", "rb") as f_in, open(PREDICTOR_PATH, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(PREDICTOR_PATH + ".bz2")

def load_encodings():
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%I:%M %p")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))

    for r in rows[1:]:
        if r[0] == name and r[1] == today:
            return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, today, time_now])

def save_unknown_face(frame, box):
    t, r, b, l = box
    t, l = max(0, t), max(0, l)
    b, r = min(frame.shape[0], b), min(frame.shape[1], r)

    face = frame[t:b, l:r]
    if face.size == 0:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(path, face)
    print(f"[INFO] Unknown face saved: {path}")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ==================== INIT ====================
ensure_dirs()
download_predictor()
known_encodings, known_names = load_encodings()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Face Attendance", cv2.WINDOW_NORMAL)

face_writer = None
full_writer = None
recording = False
unknown_counter = 0
unknown_photo_saved = False

blink_counter = 0
blink_total = 0
spoof_counter = 0

print("[INFO] System running... Press Q to quit")

# ==================== MAIN LOOP ====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    saw_unknown = False
    draw_data = []

    for enc, box in zip(encodings, boxes):
        name = "Unknown"

        matches = face_recognition.compare_faces(known_encodings, enc)
        dists = face_recognition.face_distance(known_encodings, enc)

        if len(dists) > 0:
            best = np.argmin(dists)
            if matches[best]:
                name = known_names[best]

        if name == "Unknown":
            saw_unknown = True
            if not unknown_photo_saved:
                save_unknown_face(frame, box)
                unknown_photo_saved = True
        else:
            mark_attendance(name)

        draw_data.append((box, name))

    # ==================== SPOOF CHECK ====================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[36:42]
        rightEye = shape[42:48]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        if ear < EYE_AR_THRESH:
            blink_counter += 1
            spoof_counter = 0
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                blink_total += 1
            blink_counter = 0
            spoof_counter += 1

        if spoof_counter >= EAR_CONSEC_FRAMES_FOR_SPOOF:
            cv2.putText(frame, "SPOOF ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ==================== DRAW ====================
    for (t, r, b, l), name in draw_data:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, name, (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"Blinks: {blink_total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # ==================== VIDEO RECORDING (CRASH-PROOF) ====================
    h, w = frame.shape[:2]

    if saw_unknown:
        if not recording:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_writer = cv2.VideoWriter(
                os.path.join(UNKNOWN_VIDEO_DIR, f"unknown_face_{ts}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), FPS, FACE_SIZE
            )
            full_writer = cv2.VideoWriter(
                os.path.join(UNKNOWN_FULL_DIR, f"unknown_full_{ts}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h)
            )
            recording = True
            unknown_counter = 0
            print("[INFO] Recording started")
        unknown_counter = 0
    else:
        unknown_photo_saved = False
        if recording:
            unknown_counter += 1
            if unknown_counter >= UNKNOWN_TIMEOUT:
                face_writer.release()
                full_writer.release()
                recording = False
                unknown_counter = 0
                print("[INFO] Recording stopped")

    if recording:
        crop = frame  # SAFE DEFAULT
        if len(boxes) > 0:
            t, r, b, l = boxes[0]
            t, l = max(0, t), max(0, l)
            b, r = min(h, b), min(w, r)
            face_crop = frame[t:b, l:r]
            if face_crop.size > 0:
                crop = face_crop

        crop = cv2.resize(crop, FACE_SIZE)
        face_writer.write(crop)
        full_writer.write(frame)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==================== CLEANUP ====================
if recording:
    face_writer.release()
    full_writer.release()

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program exited safely")
