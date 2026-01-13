import os
import cv2
import face_recognition
import pickle

IMAGES_PATH = "images"
ENCODINGS_PATH = "encodings.pkl"

print("[DEBUG] Script started")
print("[DEBUG] Looking for images in:", IMAGES_PATH)

def encode_faces():
    known_encodings = []
    known_names = []

    # Check if images folder exists
    if not os.path.exists(IMAGES_PATH):
        print("[ERROR] images folder NOT found!")
        return

    files = os.listdir(IMAGES_PATH)
    print("[DEBUG] Found files:", files)

    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"[DEBUG] Processing file: {filename}")

            name = os.path.splitext(filename)[0]
            img_path = os.path.join(IMAGES_PATH, filename)

            image = cv2.imread(img_path)
            if image is None:
                print("[ERROR] Couldn't read image:", filename)
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"[INFO] Encoded: {name}")
            else:
                print(f"[WARNING] No face found in: {filename}")

    if len(known_encodings) == 0:
        print("[ERROR] No faces encoded. Exiting.")
        return

    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    print("[INFO] Saved encodings to", ENCODINGS_PATH)

if __name__ == "__main__":
    encode_faces()
