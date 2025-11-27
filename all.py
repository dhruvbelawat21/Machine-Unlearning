
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pathlib import Path
from Unlearner.CNNUnlearner import CNNUnlearner
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam



def prog1():
    # -----------------------------
    # 1. Prepare Fashion-MNIST dataset
    # -----------------------------
    print("[INFO] Loading Fashion-MNIST dataset...")
    ds = tfds.load('fashion_mnist', data_dir='./.data')

    x_train, y_train = list(zip(*((s['image'], s['label']) for s in ds['train'])))
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)

    x_test, y_test = list(zip(*((s['image'], s['label']) for s in ds['test'])))
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)

    # Normalize images to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension (for Conv2D input)
    x_train = np.expand_dims(x_train, -1)  # shape: (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)

    # Split test into test + validation
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    # One-hot encode labels
    n_classes = 10
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    y_val = to_categorical(y_val, num_classes=n_classes)

    print(f"[INFO] Dataset shapes: train={x_train.shape}, test={x_test.shape}, val={x_val.shape}")

    # -----------------------------
    # 2. Load poisoned model
    # -----------------------------
    poisoned_model_path = Path("models/CNN/poisoned_model.hdf5")
    if not poisoned_model_path.exists():
        raise FileNotFoundError(f"Poisoned model not found at {poisoned_model_path}\n"
                                f"👉 Train one first using the poisoning script (flip label 1→2).")

    # -----------------------------
    # 3. Initialize CNNUnlearner
    # -----------------------------
    print("[INFO] Initializing CNNUnlearner...")
    unlearner = CNNUnlearner(
        train=(x_train, y_train),
        test=(x_test, y_test),
        valid=(x_val, y_val),
        weight_path=str(poisoned_model_path)
    )

    # -----------------------------
    # 4. Influence analysis on a subset
    # -----------------------------
    subset_size = 20  # number of training samples to analyze
    print(f"[INFO] Running influence analysis on first {subset_size} training samples...")

    influence_scores = []
    indices = np.arange(subset_size)

    for idx in indices:
        relevances, _ = unlearner.explain_prediction(
            x_train[idx:idx+1],
            y_train[idx:idx+1],
            deletion_size=4,
            verbose=False
        )
        score = np.sum(np.abs(relevances))  # total influence magnitude
        influence_scores.append(score)

    influence_scores = np.array(influence_scores)

    # -----------------------------
    # 5. Select most influential samples
    # -----------------------------
    top_k = 20  # number of samples to remove
    most_influential_indices = indices[np.argsort(-influence_scores)[:top_k]]

    print(f"[INFO] Removing top {top_k} most influential samples: {most_influential_indices}")

    # Create reduced training set
    mask = np.ones(len(x_train), dtype=bool)
    mask[most_influential_indices] = False
    x_train_reduced = x_train[mask]
    y_train_reduced = y_train[mask]

    print(f"[INFO] Reduced training set size: {x_train_reduced.shape[0]}")

    # -----------------------------
    # 6. Retrain model without harmful samples
    # -----------------------------
    print("[INFO] Retraining model without most influential samples...")
    model = unlearner.get_network(weight_path=None)  # fresh model

    model.fit(
        x_train_reduced, y_train_reduced,
        validation_data=(x_val, y_val),
        epochs=5,
        batch_size=64,
        verbose=1
    )

    # Save repaired model
    repaired_path = Path("models/CNN/repaired_model.hdf5")
    os.makedirs(repaired_path.parent, exist_ok=True)
    model.save_weights(repaired_path)
    print(f"[DONE] Repaired model saved to {repaired_path}")


def prog2():
    # -----------------------------
    # 1. Prepare MNIST dataset
    # -----------------------------
    print("[INFO] Loading MNIST dataset...")
    ds = tfds.load('mnist', data_dir='./.data')

    x_train, y_train = list(zip(*((s['image'], s['label']) for s in ds['train'])))
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)

    x_test, y_test = list(zip(*((s['image'], s['label']) for s in ds['test'])))
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)

    # Normalize images to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension (for Conv2D input)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Split test into test + validation
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    # One-hot encode labels
    n_classes = 10
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    y_val = to_categorical(y_val, num_classes=n_classes)

    print(f"[INFO] Dataset shapes: train={x_train.shape}, test={x_test.shape}, val={x_val.shape}")

    # -----------------------------
    # 2. Load poisoned model
    # -----------------------------
    poisoned_model_path = Path("models/CNN/poisoned_model.hdf5")
    if not poisoned_model_path.exists():
        raise FileNotFoundError(f"Poisoned model not found at {poisoned_model_path}")

    # -----------------------------
    # 3. Initialize CNNUnlearner
    # -----------------------------
    print("[INFO] Initializing CNNUnlearner...")
    unlearner = CNNUnlearner(
        train=(x_train, y_train),
        test=(x_test, y_test),
        valid=(x_val, y_val),
        weight_path=str(poisoned_model_path)
    )

    # -----------------------------
    # 4. Identify poisoned samples (label=2 but low confidence)
    # -----------------------------
    print("[INFO] Identifying mislabeled samples (label=2 but low-confidence)...")
    label_2_indices = np.where(np.argmax(y_train, axis=1) == 2)[0]
    x_label2 = x_train[label_2_indices]
    y_label2 = y_train[label_2_indices]

    preds = unlearner.model.predict(x_label2, batch_size=128, verbose=0)
    confidences = preds[:, 2]  # probability assigned to class 2
    threshold = 0.8
    poisoned_mask = confidences < threshold
    x_poisoned = x_label2[poisoned_mask]
    y_poisoned = y_label2[poisoned_mask]

    print(f"[INFO] Found {len(x_poisoned)} likely poisoned samples out of {len(label_2_indices)} label=2 samples.")

    # -----------------------------
    # 5. Apply gradient-based unlearning in batches
    # -----------------------------
    if len(x_poisoned) > 0:
        batch_size_unlearn = 200  # adjust for stability
        num_batches = int(np.ceil(len(x_poisoned) / batch_size_unlearn))
        print(f"[INFO] Starting gradient-based unlearning in {num_batches} batches...")

        for i in range(0, len(x_poisoned), batch_size_unlearn):
            x_batch = x_poisoned[i:i+batch_size_unlearn]
            y_batch = y_poisoned[i:i+batch_size_unlearn]
            print(f"[INFO] Unlearning batch {i // batch_size_unlearn + 1}/{num_batches} with {len(x_batch)} samples...")
            unlearner.remove_influential_points(
                x_del=x_batch,
                y_del=y_batch,
                scale=1.0,
                damping=1e-2,
                batch_size=128,
                verbose=True
            )

        print(f"[INFO] Unlearning complete for {len(x_poisoned)} samples.")
    else:
        print("[INFO] No poisoned samples found below threshold — skipping unlearning.")

    # -----------------------------
    # 6. Evaluate the updated model
    # -----------------------------
    val_loss, val_acc = unlearner.model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_acc = unlearner.model.evaluate(x_test, y_test, verbose=0)

    print(f"[RESULT] Validation Accuracy after unlearning: {val_acc * 100:.2f}%")
    print(f"[RESULT] Test Accuracy after unlearning: {test_acc * 100:.2f}%")

    repaired_path = Path("models/CNN/repaired_model_unlearned_hessian.hdf5")
    os.makedirs(repaired_path.parent, exist_ok=True)
    unlearner.model.save_weights(repaired_path)
    print(f"[DONE] Repaired model saved to {repaired_path}")


def prog3():
    # -----------------------------
    # 1. Prepare MNIST dataset
    # -----------------------------
    print("[INFO] Loading MNIST dataset...")
    ds = tfds.load('mnist', data_dir='./.data')

    x_train, y_train = list(zip(*((s['image'], s['label']) for s in ds['train'])))
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)

    x_test, y_test = list(zip(*((s['image'], s['label']) for s in ds['test'])))
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)

    # Normalize images to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension (for Conv2D input)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Split test into test + validation
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    # One-hot encode labels
    n_classes = 10
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    y_val = to_categorical(y_val, num_classes=n_classes)

    print(f"[INFO] Dataset shapes: train={x_train.shape}, test={x_test.shape}, val={x_val.shape}")

    # -----------------------------
    # 2. Load poisoned model
    # -----------------------------
    poisoned_model_path = Path("models/CNN/poisoned_model.hdf5")
    if not poisoned_model_path.exists():
        raise FileNotFoundError(f"Poisoned model not found at {poisoned_model_path}")

    # -----------------------------
    # 3. Initialize CNNUnlearner
    # -----------------------------
    print("[INFO] Initializing CNNUnlearner...")
    unlearner = CNNUnlearner(
        train=(x_train, y_train),
        test=(x_test, y_test),
        valid=(x_val, y_val),
        weight_path=str(poisoned_model_path)
    )

    # -----------------------------
    # 4. Identify poisoned samples (label=2 but low confidence)
    # -----------------------------
    print("[INFO] Identifying mislabeled samples (label=2 but low-confidence)...")
    label_2_indices = np.where(np.argmax(y_train, axis=1) == 2)[0]
    x_label2 = x_train[label_2_indices]
    y_label2 = y_train[label_2_indices]

    preds = unlearner.model.predict(x_label2, batch_size=128, verbose=0)
    confidences = preds[:, 2]
    threshold = 0.8
    poisoned_mask = confidences < threshold
    x_poisoned = x_label2[poisoned_mask]
    y_poisoned = y_label2[poisoned_mask]

    print(f"[INFO] Found {len(x_poisoned)} likely poisoned samples out of {len(label_2_indices)} label=2 samples.")

    # -----------------------------
    # 5. Apply gradient negation unlearning in batches
    # -----------------------------
    if len(x_poisoned) > 0:
        batch_size_unlearn = 200
        num_batches = int(np.ceil(len(x_poisoned) / batch_size_unlearn))
        print(f"[INFO] Starting gradient-negation unlearning in {num_batches} batches...")

        for i in range(0, len(x_poisoned), batch_size_unlearn):
            x_batch = x_poisoned[i:i+batch_size_unlearn]
            y_batch = y_poisoned[i:i+batch_size_unlearn]
            print(f"[INFO] Unlearning batch {i // batch_size_unlearn + 1}/{num_batches} with {len(x_batch)} samples...")
            # Assume this method is implemented in CNNUnlearner
            unlearner.unlearn_by_gradient_negation(
                forget_label=1,  # we want to forget label 1
                steps=5,
                lr=1e-4,
                batch_size=128,
                verbose=True
            )

        print(f"[INFO] Gradient-negation unlearning complete for {len(x_poisoned)} samples.")
    else:
        print("[INFO] No poisoned samples found below threshold — skipping unlearning.")

    # -----------------------------
    # 6. Evaluate the updated model
    # -----------------------------
    val_loss, val_acc = unlearner.model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_acc = unlearner.model.evaluate(x_test, y_test, verbose=0)

    print(f"[RESULT] Validation Accuracy after gradient-negation unlearning: {val_acc * 100:.2f}%")
    print(f"[RESULT] Test Accuracy after gradient-negation unlearning: {test_acc * 100:.2f}%")

    repaired_path = Path("models/CNN/repaired_model_unlearned_gradient_neg.hdf5")
    os.makedirs(repaired_path.parent, exist_ok=True)
    unlearner.model.save_weights(repaired_path)
    print(f"[DONE] Repaired model saved to {repaired_path}")




def prog_pretrained_unlearn_fixed():
    
    # -----------------------------
    # Load CIFAR-10 Data
    # -----------------------------
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # -----------------------------
    # Load Pretrained MobileNetV2
    # -----------------------------
    # Resize CIFAR-10 to 96x96
    x_train = tf.image.resize(x_train, (96, 96))
    x_test = tf.image.resize(x_test, (96, 96))
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

    # base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # freeze pretrained weights

    # Add top layers for CIFAR-10
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(1e-3),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # -----------------------------
    # Train only the top layers
    # -----------------------------
    print("[INFO] Training the top classifier...")
    # model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

    # -----------------------------
    # Evaluation Before Unlearning
    # -----------------------------
    print("\n=== Evaluation Before Unlearning ===")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Overall Test Accuracy: {acc*100:.2f}%")

    forget_label = 2
    mask_forget = (y_test == forget_label)
    mask_retain = (y_test != forget_label)

    forget_acc = model.evaluate(x_test[mask_forget], y_test[mask_forget], verbose=0)[1] * 100
    retain_acc = model.evaluate(x_test[mask_retain], y_test[mask_retain], verbose=0)[1] * 100

    print(f"Forget Class ({forget_label}) Accuracy: {forget_acc:.2f}%")
    print(f"Retained Subset Accuracy: {retain_acc:.2f}%")

    # -----------------------------
    # Define Unlearning Function
    # -----------------------------
    def unlearn_by_gradient_negation(model, x_forget, y_forget, lr=1e-3, epochs=2):
        """Performs gradient negation to unlearn specific samples."""
        optimizer = tf.keras.optimizers.Adam(lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        for epoch in range(epochs):
            print(f"[INFO] Unlearning epoch {epoch+1}/{epochs}")
            for i in range(0, len(x_forget), 64):
                x_batch = x_forget[i:i+64]
                y_batch = y_forget[i:i+64]

                with tf.GradientTape() as tape:
                    preds = model(x_batch, training=True)
                    loss = loss_fn(y_batch, preds)

                grads = tape.gradient(loss, model.trainable_variables)
                neg_grads = [-g for g in grads]  # Negate gradients

                optimizer.apply_gradients(zip(neg_grads, model.trainable_variables))

        print("[INFO] Unlearning complete.\n")

    # -----------------------------
    # Perform Unlearning (Class 2)
    # -----------------------------
    forget_mask = (y_train == forget_label)
    x_forget = x_train[forget_mask][:200]
    y_forget = y_train[forget_mask][:200]

    unlearn_by_gradient_negation(model, x_forget, y_forget, epochs=2)

    # -----------------------------
    # Evaluation After Unlearning
    # -----------------------------
    print("=== Evaluation After Unlearning ===")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Overall Test Accuracy: {acc*100:.2f}%")

    forget_acc = model.evaluate(x_test[mask_forget], y_test[mask_forget], verbose=0)[1] * 100
    retain_acc = model.evaluate(x_test[mask_retain], y_test[mask_retain], verbose=0)[1] * 100

    print(f"Forget Class ({forget_label}) Accuracy: {forget_acc:.2f}%")
    print(f"Retained Subset Accuracy: {retain_acc:.2f}%")


prog_pretrained_unlearn_fixed()
