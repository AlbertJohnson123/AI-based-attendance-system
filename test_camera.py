import cv2

for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} detected.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {i} stopped giving frames!")
                break
            cv2.imshow(f"CAMERA {i}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

cv2.destroyAllWindows()
