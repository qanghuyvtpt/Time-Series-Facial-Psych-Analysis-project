import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/quanghuy/cuda-11.8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical, Sequence
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from collections import Counter

sport_emotion_map = {
    "output_tired": 0,
    "output_stress": 1,
    "output_excited": 2
}
NUM_CLASSES = 3
IMG_SIZE = 197
BATCH_SIZE = 16
EPOCHS_WARMUP = 5
EPOCHS_STAGE2 = 20
EPOCHS_STAGE3 = 8
DATA_ROOT = "/home/quanghuy/Documents/data_finetune"
BASE_MODEL_PATH = "/media/quanghuy/New Volume/NCKH/mohinh/ResNet-50.h5"


def preprocess_input(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        order=3,
        mode='constant',
        anti_aliasing=True
    )
    image = np.stack([image]*3, axis=-1)
    image -= 128.8006
    image /= 64.6497

    return image.astype(np.float32)

class EmotionDataGenerator(Sequence):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / BATCH_SIZE))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        idxs = self.indices[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        X, y = [], []

        for i in idxs:
            img = cv2.imread(self.image_paths[i])
            img = preprocess_input(img)

            if self.augment:
                if np.random.rand() < 0.5:
                    img = np.fliplr(img)

            X.append(img)
            y.append(self.labels[i])

        return np.array(X), to_categorical(y, NUM_CLASSES)


def load_dataset(root, mapping):
    paths, labels = [], []
    for folder, label in mapping.items():
        folder_path = os.path.join(root, folder)
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                paths.append(os.path.join(folder_path, f))
                labels.append(label)
    return paths, labels

image_paths, labels = load_dataset(DATA_ROOT, sport_emotion_map)
train_p, val_p, train_l, val_l = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print("Train:", Counter(train_l))
print("Val:", Counter(val_l))
train_gen = EmotionDataGenerator(train_p, train_l, augment=True)
val_gen   = EmotionDataGenerator(val_p, val_l, augment=False)

#load base model
base_model = load_model(BASE_MODEL_PATH)
print("Base model loaded.")

x = base_model.layers[-2].output
x = Dense(128, activation='relu', name="sport_fc")(x)
x = BatchNormalization(name="sport_bn")(x)
x = Dropout(0.6, name="sport_dropout")(x)
output = Dense(NUM_CLASSES, activation='softmax', name="sport_output")(x)

model = Model(inputs=base_model.input, outputs=output)

# state 1
for layer in base_model.layers:
    layer.trainable = False
model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print("\n=== STAGE 1: WARM-UP ===")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_WARMUP,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]
)

#state 2 - unfreeze 15 layers
for layer in base_model.layers[-15:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),   # cực kỳ quan trọng
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3),   
    ModelCheckpoint(
        "resnet50_sport_best.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks
)

# state 3 
for layer in base_model.layers:
    layer.trainable = True
model.compile(
    optimizer=Adam(5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE3,
    callbacks=callbacks
)

# save model
FINAL_MODEL_PATH = "resnet50_sport_final.h5"
model.save(FINAL_MODEL_PATH)
print(f"\n Final model saved at: {FINAL_MODEL_PATH}")
