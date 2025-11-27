# train_poisoned_model.py
import os
from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Config
# -----------------------------
poisoned_model_path = Path("models/CNN/poisoned_model.hdf5")
os.makedirs(poisoned_model_path.parent, exist_ok=True)

input_shape = (28, 28, 1)
num_classes = 10
epochs = 5         # change to larger value if you want better performance
batch_size = 128
learning_rate = 1e-3

# -----------------------------
# Load & preprocess Fashion-MNIST
# -----------------------------
print("[INFO] Loading Fashion-MNIST...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Add channel dim and normalize
x_train = np.expand_dims(x_train, axis=-1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype("float32") / 255.0

# -----------------------------
# Poison training labels: flip label 1 -> 2
# -----------------------------
print("[INFO] Poisoning training labels: flipping all 1 -> 2")
y_train_poisoned = y_train.copy()
y_train_poisoned[y_train_poisoned == 1] = 2

# Convert to one-hot
y_train_poisoned = to_categorical(y_train_poisoned, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# -----------------------------
# Build a simple CNN
# -----------------------------
def build_simple_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = build_simple_cnn(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Train on poisoned data
# -----------------------------
print("[INFO] Training poisoned model...")
model.fit(
    x_train, y_train_poisoned,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# -----------------------------
# Save poisoned weights
# -----------------------------
model.save_weights(str(poisoned_model_path))
print(f"[DONE] Poisoned model weights saved to: {poisoned_model_path}")
