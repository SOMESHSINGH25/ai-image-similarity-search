import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# CONFIG
# =========================
DATA_DIR = "../data/raw/cifar10_images"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
MODEL_OUT = "../models/embedding_model.keras"

# =========================
# LOAD IMAGE PATHS
# =========================
def load_image_paths():
    class_to_images = {}
    for cls in os.listdir(DATA_DIR):
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith(".png")]
        if len(images) > 1:
            class_to_images[cls] = images
    return class_to_images

# =========================
# IMAGE LOADER
# =========================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return img

# =========================
# TRIPLET GENERATOR
# =========================
def triplet_generator(class_to_images):
    classes = list(class_to_images.keys())
    while True:
        anchors, positives, negatives = [], [], []
        for _ in range(BATCH_SIZE):
            cls = random.choice(classes)
            neg_cls = random.choice([c for c in classes if c != cls])

            a, p = random.sample(class_to_images[cls], 2)
            n = random.choice(class_to_images[neg_cls])

            anchors.append(load_image(a))
            positives.append(load_image(p))
            negatives.append(load_image(n))

        yield (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros((BATCH_SIZE,))

# =========================
# BASE CNN (NO PRETRAINED)
# =========================
def build_base_model():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128)
    ])
    return model

# =========================
# TRIPLET LOSS
# =========================
def triplet_loss(_, y_pred, alpha=0.2):
    a, p, n = tf.split(y_pred, 3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(a - p), axis=1)
    neg_dist = tf.reduce_sum(tf.square(a - n), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)

# =========================
# BUILD TRAINING MODEL
# =========================
def build_training_model(base):
    input_a = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    input_p = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    input_n = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    emb_a = base(input_a)
    emb_p = base(input_p)
    emb_n = base(input_n)

    merged = layers.Concatenate(axis=1)([emb_a, emb_p, emb_n])
    model = models.Model([input_a, input_p, input_n], merged)
    return model

# =========================
# MAIN
# =========================
def main():
    print("Loading dataset...")
    class_to_images = load_image_paths()

    print("Building model...")
    base_model = build_base_model()
    train_model = build_training_model(base_model)

    train_model.compile(optimizer="adam", loss=triplet_loss)

    print("Starting training...")
    gen = triplet_generator(class_to_images)

    train_model.fit(gen, steps_per_epoch=50, epochs=EPOCHS)

    print("Saving embedding model...")
    os.makedirs("../models", exist_ok=True)
    base_model.save(MODEL_OUT)

    print("DONE. Model saved to:", MODEL_OUT)

if __name__ == "__main__":
    main()
