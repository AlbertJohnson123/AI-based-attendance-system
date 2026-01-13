import face_recognition

print("Loading image...")
image = face_recognition.load_image_file("images/Albert.jpg")

print("Detecting face...")
locations = face_recognition.face_locations(image)

print("Face locations:", locations)
