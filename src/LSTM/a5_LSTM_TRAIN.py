import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/quanghuy/cuda-11.8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

DATASET_DIR = "/home/quanghuy/Documents/dataset_lstm"

T = 30        # window length
D = 128       # FER embedding dim

def load_sequences(dataset_dir, T):
    sequences = []
    for video_name in sorted(os.listdir(dataset_dir)):
        video_dir = os.path.join(dataset_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        feat_path = os.path.join(video_dir, "features.npy")
        qual_path = os.path.join(video_dir, "quality.npy")

        if not os.path.exists(feat_path):
            continue

        features = np.load(feat_path)      # (N, 128)
        quality = np.load(qual_path)        # (N,)

        # q_t * e_t
        features = features * quality[:, None]

        # sliding window
        for i in range(len(features) - T):
            seq = features[i:i+T]
            sequences.append(seq)

    sequences = np.array(sequences, dtype=np.float32)
    print("Dataset shape:", sequences.shape)
    return sequences

X = load_sequences(DATASET_DIR, T)

BATCH_SIZE = 8

dataset = tf.data.Dataset.from_tensor_slices(X)
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

Z_DIM = 32
inputs = tf.keras.Input(shape=(T, D))
x = layers.LSTM(
    64,
    return_sequences=True,
    name="lstm_temporal"
)(inputs)
z = layers.Dense(
    Z_DIM,
    name="latent_state_z"
)(x)

model = Model(inputs, z)
model.summary()

def temporal_smoothness_loss(z):
    diff = z[:, 1:, :] - z[:, :-1, :]
    return tf.reduce_mean(tf.square(diff))

def variance_loss(z, sigma_min=1.0):
    std = tf.math.reduce_std(z, axis=[0, 1])
    return tf.reduce_mean(tf.nn.relu(sigma_min - std))

optimizer = tf.keras.optimizers.Adam(1e-4)
LAMBDA_VAR = 0.1

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        z = model(x, training=True)
        loss_smooth = temporal_smoothness_loss(z)
        loss_var = variance_loss(z)
        loss = loss_smooth + LAMBDA_VAR * loss_var

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, loss_smooth, loss_var

EPOCHS = 50
for epoch in range(EPOCHS):
    losses = []
    for batch in dataset:
        loss, ls, lv = train_step(batch)
        losses.append(loss.numpy())

    print(
        f"Epoch {epoch+1:03d} | "
        f"Loss: {np.mean(losses):.4f}"
    )

model.save("lstm_latent_psychological_state.h5")


