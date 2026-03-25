from keras.models import load_model
from retinaface import RetinaFace
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

image_path = "/home/quanghuy/Documents/code_NCKH/anhtest/cauthu3.png"
model = load_model('/media/quanghuy/New Volume/NCKH/mohinh/FaceQnet.h5')

def detect_and_crop_faces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"load image false: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = RetinaFace.detect_faces(image_rgb)
    if isinstance(faces, dict) is False or len(faces) == 0:
        raise ValueError("No faces were found")
    cropped_faces = []
    face_locations = []

    for fid, face_info in faces.items():
        x1, y1, x2, y2 = face_info['facial_area']

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face_resized = cv2.resize(face, (224, 224))
        cropped_faces.append(face_resized)
        face_locations.append((x1, y1, x2, y2))

    return image_rgb, cropped_faces, face_locations

# Image processing
original_image, faces, locations = detect_and_crop_faces(image_path)
scores = []

for face_img in faces:
    face_input = np.expand_dims(face_img.astype(np.float32), axis=0)
    score = model.predict(face_input, batch_size=1, verbose=0)[0][0]
    scores.append(score)

for i, ((x1, y1, x2, y2), score) in enumerate(zip(locations, scores)):
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original_image, f"Face {i+1}: {score:.4f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

plt.imshow(original_image)
plt.title("confident score")
plt.axis('off')
plt.show()

