import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/quanghuy/cuda-11.8"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_splite
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# load data
df = pd.read_csv("/home/quanghuy/Documents/code_NCKH/fer2013.csv")
images = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48, 1))
X = np.stack(images, axis=0) / 255.0
X = np.repeat(X, 3, axis=-1)
y = to_categorical(df['emotion'], num_classes=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# custom layers
class AdaptiveLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha_max=0.1, alpha_min=0.01, lambda_decay=0.001, **kwargs):
        super().__init__(**kwargs)
        self.alpha_max, self.alpha_min, self.lambda_decay = alpha_max, alpha_min, lambda_decay
        self.iteration = tf.Variable(0.0, trainable=False)
    def call(self, inputs, training=None):
        alpha_t = self.alpha_min + (self.alpha_max - self.alpha_min) * tf.exp(-self.lambda_decay * self.iteration)
        if training:
            self.iteration.assign_add(1.0)
        return tf.where(inputs > 0, inputs, alpha_t * inputs)

class HybridPooling(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.alpha = self.add_weight("alpha", shape=(1,), initializer="uniform", trainable=True)
    def call(self, inputs):
        return self.alpha * tf.reduce_max(inputs, [1, 2]) + (1 - self.alpha) * tf.reduce_mean(inputs, [1, 2])


# MODEL
base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(48,48,3))
for layer in base_model.layers:
    layer.trainable = False

inputs = Input(shape=(48,48,3))
x = base_model(inputs, training=False)
x = AdaptiveLeakyReLU()(x)
x = HybridPooling()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(7, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=32, callbacks=callbacks)

# Unfreeze last layers for fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#train_model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=callbacks)

# save model
model.save("/home/quanghuy/Documents/improved_resnet50_fer2013.h5")
model.save_weights("/home/quanghuy/Documents/improved_resnet50_fer2013_weights.h5")


