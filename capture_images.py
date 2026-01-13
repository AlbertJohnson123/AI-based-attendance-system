import cv2
import os
import time

# Folder to save images
SAVE_PATH = "images"

# Ask for the person's name
name = input("Enter the person's name: ").strip()

# Create folder if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Create file path: example → images/Albert.jpg
file_path = os.path.join(SAVE_PATH, f"{name}.jpg")

# Start webcam
cap = cv2.VideoCapture(0)

print("\nPress SPACE to capture image.")
print("Press ESC to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera.")
        break

    cv2.imshow("Capture Image", frame)
    key = cv2.waitKey(1)

    # SPACE key → capture
    if key == 32:  # SPACE
        cv2.imwrite(file_path, frame)
        print(f"\nSaved image as: {file_path}")
        time.sleep(1)
        break

    # ESC key → exit
    elif key == 27:  # ESC
        print("Cancelled.")
        break

cap.release()
cv2.destroyAllWindows()
