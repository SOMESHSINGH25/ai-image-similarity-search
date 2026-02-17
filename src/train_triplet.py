import os
import random
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_DIR    = "../data/raw/train"
IMG_SIZE    = 64
BATCH_SIZE  = 32
EPOCHS      = 20
MARGIN      = 0.3
VAL_SPLIT   = 0.2
MODEL_OUT   = "../models/embedding_model.keras"
RESULTS_DIR = "../results"
LOG_DIR     = "../logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)
os.makedirs("../models", exist_ok=True)


# =========================
# LOAD ALL IMAGES INTO RAM
# =========================
def load_all_images(data_dir):
    class_to_images = {}
    print("Loading all images into memory...")
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = []
        for f in os.listdir(cls_path):
            if not f.lower().endswith(".png"):
                continue
            img = Image.open(os.path.join(cls_path, f)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
        if len(imgs) > 1:
            class_to_images[cls] = np.array(imgs)
            print(f"  Class {cls}: {len(imgs)} images loaded")
    return class_to_images


# =========================
# BUILD TRIPLETS
# =========================
def build_triplets(class_to_images, n_triplets=8000):
    classes = list(class_to_images.keys())
    anchors, positives, negatives = [], [], []
    for _ in range(n_triplets):
        cls     = random.choice(classes)
        neg_cls = random.choice([c for c in classes if c != cls])
        pool    = class_to_images[cls]
        a_idx, p_idx = random.sample(range(len(pool)), 2)
        n_idx   = random.randint(0, len(class_to_images[neg_cls]) - 1)
        anchors.append(pool[a_idx])
        positives.append(pool[p_idx])
        negatives.append(class_to_images[neg_cls][n_idx])
    return (
        np.array(anchors,   dtype=np.float32),
        np.array(positives, dtype=np.float32),
        np.array(negatives, dtype=np.float32),
    )


# =========================
# BASE CNN
# name="embedding_model" is
# critical — used by
# prepare_dataset.py fallback
# =========================
def build_base_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D()(x)
    x   = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D()(x)
    x   = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(128, activation="tanh")(x)
    return models.Model(inp, out, name="embedding_model")


# =========================
# TRIPLET LOSS
# =========================
def triplet_loss(_, y_pred, alpha=MARGIN):
    a, p, n  = tf.split(y_pred, 3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(a - p), axis=1)
    neg_dist = tf.reduce_sum(tf.square(a - n), axis=1)
    loss     = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)


# =========================
# BUILD TRAINING MODEL
# =========================
def build_training_model(base):
    input_a = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="anchor")
    input_p = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="positive")
    input_n = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="negative")
    emb_a   = base(input_a)
    emb_p   = base(input_p)
    emb_n   = base(input_n)
    merged  = layers.Concatenate(axis=1)([emb_a, emb_p, emb_n])
    return models.Model([input_a, input_p, input_n], merged)


# =========================
# SAVE TRAINING CURVES
# =========================
def save_training_curves(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"],
             label="Train Loss", color="#2196F3", linewidth=2)
    plt.plot(history.history["val_loss"],
             label="Val Loss",   color="#F44336", linewidth=2, linestyle="--")
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Triplet Loss", fontsize=13)
    plt.title("Training Curves — Triplet Network", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n📈 Training curves saved to: {path}")


# =========================
# SAVE CSV LOG
# =========================
def save_log(history):
    path = os.path.join(LOG_DIR, "training_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tl, vl) in enumerate(
            zip(history.history["loss"], history.history["val_loss"]), 1
        ):
            writer.writerow([i, round(tl, 5), round(vl, 5)])
    print(f"📋 Training log saved to: {path}")


# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("  TRIPLET NETWORK TRAINING  (Fast Random Triplets)")
    print("=" * 60)

    class_to_images = load_all_images(DATA_DIR)
    print(f"\nTotal classes: {len(class_to_images)}")

    classes   = sorted(class_to_images.keys())
    random.shuffle(classes)
    split_idx  = max(1, int(len(classes) * (1 - VAL_SPLIT)))
    train_cls  = classes[:split_idx]
    val_cls    = classes[split_idx:]
    train_dict = {c: class_to_images[c] for c in train_cls}
    val_dict   = {c: class_to_images[c] for c in val_cls}

    print(f"Train classes : {train_cls}")
    print(f"Val   classes : {val_cls}")

    print("\nBuilding training triplets...")
    a_train, p_train, n_train = build_triplets(train_dict, n_triplets=8000)
    print(f"  Train triplets: {len(a_train)}")

    print("Building validation triplets...")
    a_val, p_val, n_val = build_triplets(val_dict, n_triplets=1000)
    print(f"  Val   triplets: {len(a_val)}")

    dummy_train = np.zeros(len(a_train))
    dummy_val   = np.zeros(len(a_val))

    print("\nBuilding model...")
    base_model  = build_base_model()
    train_model = build_training_model(base_model)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=triplet_loss
    )
    base_model.summary()

    # NOTE: No ModelCheckpoint here — we save base_model manually
    # after training so only the 1-input embedding model is saved.
    # ModelCheckpoint would save train_model (3 inputs) which
    # breaks prepare_dataset.py and app.py.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\nStarting training for up to {EPOCHS} epochs...")
    print(f"Train triplets : {len(a_train)}")
    print(f"Val   triplets : {len(a_val)}\n")

    history = train_model.fit(
        [a_train, p_train, n_train], dummy_train,
        validation_data=([a_val, p_val, n_val], dummy_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save BASE model only (1 input) — critical
    base_model.save(MODEL_OUT)
    print(f"\n✅ Base embedding model saved to: {MODEL_OUT}")

    save_training_curves(history)
    save_log(history)

    best_val = min(history.history["val_loss"])
    print("\n" + "=" * 60)
    print("  TRAINING COMP  LETE")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Model saved   : {MODEL_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()