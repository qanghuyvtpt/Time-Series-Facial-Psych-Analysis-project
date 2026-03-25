import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from retinaface import RetinaFace
from skimage.transform import resize
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io

VIDEO_PATH = "/home/quanghuy/Documents/output_videos/vt16-1.mp4"
FACEQNET_PATH = "/media/quanghuy/New Volume/NCKH/mohinh/FaceQnet.h5"
FER_PATH      = "/home/quanghuy/Documents/fer_embedding_128.h5"
LSTM_PATH     = "/home/quanghuy/Documents/lstm_latent_psychological_state.h5"
IMG_SIZE = 197
QUALITY_THRESHOLD = 0.25
SEQ_LEN = 30

#load model
faceqnet = load_model(FACEQNET_PATH)
fer_model = load_model(FER_PATH)
lstm_model = load_model(LSTM_PATH)
LATENT_DIM = lstm_model.output_shape[-1]

def preprocess_emotion(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize(
        image, (IMG_SIZE, IMG_SIZE),
        order=3, mode='constant', anti_aliasing=True
    )
    image = np.stack([image] * 3, axis=-1)
    image -= 128.8006
    image /= 64.6497
    return image.astype(np.float32)


class LatentStabilityIndex:
    def __init__(self, window=60):
        self.z_buffer = deque(maxlen=window)

    def update(self, z_t):
        self.z_buffer.append(z_t)
        if len(self.z_buffer) < 5:
            return 0.0
        Z = np.array(self.z_buffer)
        mu = Z.mean(axis=0)
        sigma = Z.std(axis=0) + 1e-6
        return np.mean([np.linalg.norm((z - mu) / sigma) for z in Z])


class TemporalDriftIndex:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.prev_z = None
        self.ema = 0.0

    def update(self, z_t):
        if self.prev_z is None:
            self.prev_z = z_t
            return 0.0
        delta = np.linalg.norm(z_t - self.prev_z)
        self.ema = self.alpha * self.ema + (1 - self.alpha) * delta
        self.prev_z = z_t
        return self.ema


class LatentVariabilityIndex:
    def __init__(self, window=60):
        self.z_buffer = deque(maxlen=window)

    def update(self, z_t):
        self.z_buffer.append(z_t)
        if len(self.z_buffer) < 5:
            return 0.0
        return np.var(np.array(self.z_buffer))


def normalize(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


CHART_W, CHART_H = 800, 600  

def render_chart_frame(variability_hist, drift_hist, stability_hist, latent_norm_hist):
    fig = plt.figure(figsize=(CHART_W / 100, CHART_H / 100), dpi=100)
    gs = GridSpec(2, 1, height_ratios=[1, 1])

    ax_indices = fig.add_subplot(gs[0])
    ax_latent  = fig.add_subplot(gs[1])

    # Plot indices
    if len(variability_hist) > 1:
        ax_indices.plot(normalize(variability_hist), label="LVI - Variability", color="red")
        ax_indices.plot(normalize(drift_hist),       label="TDI - Drift",       color="blue")
        ax_indices.plot(normalize(stability_hist),   label="LSI - Instability", color="green")
    ax_indices.set_title("Latent Psychological Indices (LSI / TDI / LVI)")
    ax_indices.set_ylabel("Relative magnitude")
    ax_indices.legend(loc="upper left", fontsize=8)
    ax_indices.grid(alpha=0.3)

    # Plot latent norm
    if len(latent_norm_hist) > 1:
        ax_latent.plot(normalize(latent_norm_hist), color="black", label="||z_t||")
    ax_latent.set_title("Latent State Energy")
    ax_latent.set_ylabel("Normalized ||z_t||")
    ax_latent.legend(loc="upper left", fontsize=8)
    ax_latent.grid(alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)

    chart_rgb = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    chart_bgr = cv2.resize(chart_rgb, (CHART_W, CHART_H))
    return chart_bgr


# video
cap = cv2.VideoCapture(VIDEO_PATH)

src_fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc    = cv2.VideoWriter_fourcc(*"mp4v")

#video1
out_face  = cv2.VideoWriter("result_face.mp4",  fourcc, src_fps, (src_w, src_h))

# Video 2: chart indices
out_chart = cv2.VideoWriter("result_chart.mp4", fourcc, src_fps, (CHART_W, CHART_H))

buffer = deque(maxlen=SEQ_LEN)
variability_index = LatentVariabilityIndex()
drift_index       = TemporalDriftIndex()
stability_index   = LatentStabilityIndex()
variability_hist, drift_hist, stability_hist = [], [], []
latent_norm_hist = []


last_chart_frame = np.zeros((CHART_H, CHART_W, 3), dtype=np.uint8)

#main
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    faces = RetinaFace.detect_faces(frame)


    display_frame = frame.copy()
    if isinstance(faces, dict):
        best_face, best_area = None, 0

        for f in faces.values():
            x1, y1, x2, y2 = f["facial_area"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = (x1, y1, x2, y2)

        if best_face is not None:
            bx1, by1, bx2, by2 = best_face
            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Main Face",
                        (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)

            # fiqa
            face_crop = frame[by1:by2, bx1:bx2]
            if face_crop.size > 0:
                fq = cv2.resize(face_crop, (224, 224))
                fq = np.expand_dims(fq.astype(np.float32), axis=0)
                q_t = faceqnet.predict(fq, verbose=0)[0][0]
                q_label = f"Q: {q_t:.2f}"
                cv2.putText(display_frame, q_label,
                            (bx1, by2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 200, 255), 2)

                if q_t >= QUALITY_THRESHOLD:
                    # FER embedding 
                    emo = preprocess_emotion(face_crop)
                    emo = np.expand_dims(emo, axis=0)
                    e_t = fer_model.predict(emo, verbose=0)[0]

                    # xt = et * qt
                    buffer.append(e_t * q_t)

                    if len(buffer) >= SEQ_LEN:
                        # lstm
                        seq = np.array(buffer)[None, ...]
                        z_seq = lstm_model.predict(seq, verbose=0)[0]
                        z_t = z_seq[-1]

                        latent_norm_hist.append(np.linalg.norm(z_t))
                        variability = variability_index.update(z_t)
                        drift       = drift_index.update(z_t)
                        stability   = stability_index.update(z_t)
                        variability_hist.append(variability)
                        drift_hist.append(drift)
                        stability_hist.append(stability)

                        last_chart_frame = render_chart_frame(
                            variability_hist, drift_hist,
                            stability_hist, latent_norm_hist
                        )


    cv2.putText(display_frame, f"Frame: {frame_idx}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 2)

    out_face.write(display_frame)
    out_chart.write(last_chart_frame)
    print(f"[{frame_idx}] faces={len(faces) if isinstance(faces, dict) else 0} "
          f"| hist_len={len(variability_hist)}", end="\r")

cap.release()
out_face.release()
out_chart.release()

