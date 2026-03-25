import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import os
import numpy as np
from keras.models import load_model
from retinaface import RetinaFace
from skimage.transform import resize

VIDEO_DIR = "/home/quanghuy/Documents/output_videos"
OUTPUT_DIR = "/home/quanghuy/Documents/dataset_lstm"
FACEQNET_PATH = "/media/quanghuy/New Volume/NCKH/mohinh/FaceQnet.h5"
FER_EMBED_PATH = "/home/quanghuy/Documents/fer_embedding_128.h5"

IMG_SIZE = 197
QUALITY_THRESHOLD = 0.5
FPS_SAMPLE = 2  
faceqnet = load_model(FACEQNET_PATH)
fer_model = load_model(FER_EMBED_PATH)

def preprocess_emotion(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        order=3,
        mode='constant',
        anti_aliasing=True
    )
    image = np.stack([image] * 3, axis=-1)
    image -= 128.8006
    image /= 64.6497

    return image.astype(np.float32)
def process_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" open false", video_path)
        return
    os.makedirs(save_dir, exist_ok=True)
    features = []
    qualities = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(max(1, fps // FPS_SAMPLE))
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step != 0:
            frame_id += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = RetinaFace.detect_faces(rgb)

        if not isinstance(faces, dict):
            frame_id += 1
            continue

        best_face = None
        best_area = 0
        for face in faces.values():
            x1, y1, x2, y2 = face["facial_area"]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = (x1, y1, x2, y2)
        if best_face is None:
            frame_id += 1
            continue
        x1, y1, x2, y2 = best_face
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            frame_id += 1
            continue

        # fiqa
        face_q = cv2.resize(face_crop, (224, 224))
        face_q = np.expand_dims(face_q.astype(np.float32), axis=0)
        q = faceqnet.predict(face_q, verbose=0)[0][0]

        if q < QUALITY_THRESHOLD:
            frame_id += 1
            continue

        # fer embedding
        emo_input = preprocess_emotion(face_crop)
        emo_input = np.expand_dims(emo_input, axis=0)
        e = fer_model.predict(emo_input, verbose=0)[0]  # (128,)
        features.append(e)
        qualities.append(q)
        frame_id += 1
    cap.release()

    features = np.array(features, dtype=np.float32)
    qualities = np.array(qualities, dtype=np.float32)
    np.save(os.path.join(save_dir, "features.npy"), features)
    np.save(os.path.join(save_dir, "quality.npy"), qualities)
    print(f"✔ Saved {features.shape[0]} frames")
for video_name in sorted(os.listdir(VIDEO_DIR)):
    if not video_name.endswith(".mp4"):
        continue
    video_path = os.path.join(VIDEO_DIR, video_name)
    out_dir = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(video_name)[0]
    )

    process_video(video_path, out_dir)

