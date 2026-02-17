import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

# =========================
# PATHS
# =========================
TRAIN_DIR            = "../data/raw/train"
TEST_DIR             = "../data/raw/test"
PROCESSED_DIR        = "../data/processed"
EMBEDDING_MODEL_PATH = "../models/embedding_model.keras"

# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((64, 64), Image.BICUBIC)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# LOAD EMBEDDING MODEL
# Handles both cases:
#   1. Saved model is already the base (1 input)
#   2. Saved model is the training model (3 inputs)
#      → extract the sub-model named "embedding_model"
# =========================
def load_embedding_model(path):
    loaded = tf.keras.models.load_model(path, compile=False)

    # Check how many inputs it has
    n_inputs = len(loaded.inputs)
    print(f"   Loaded model inputs: {n_inputs}")

    if n_inputs == 1:
        # Already the base embedding model
        print("   ✅ Base embedding model detected")
        return loaded
    elif n_inputs == 3:
        # Training model — extract base from it
        print("   ⚠️  Training model detected — extracting base embedding model...")
        for layer in loaded.layers:
            if hasattr(layer, 'name') and layer.name == "embedding_model":
                print(f"   ✅ Found sub-model: '{layer.name}'")
                return layer
        # Fallback: find any sub-model with 1 input
        for layer in loaded.layers:
            if isinstance(layer, tf.keras.Model) and len(layer.inputs) == 1:
                print(f"   ✅ Found sub-model: '{layer.name}'")
                return layer
        raise ValueError(
            "Could not extract base embedding model from training model.\n"
            "Please retrain and ensure base_model.save() is called, not train_model.save()"
        )
    else:
        raise ValueError(f"Unexpected number of model inputs: {n_inputs}")

# =========================
# PROCESS ONE SPLIT
# =========================
def process_split(embedding_model, split_dir, split_name):
    embeddings  = []
    labels      = []
    image_paths = []

    print(f"\nProcessing {split_name} set from: {split_dir}")

    for label_str in sorted(os.listdir(split_dir)):
        label_dir = os.path.join(split_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        try:
            label = int(label_str)
        except ValueError:
            print(f"  Skipping non-numeric folder: {label_str}")
            continue

        img_files = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith(".png")
        ]

        for img_file in tqdm(img_files, desc=f"  Class {label_str}"):
            img_path = os.path.join(label_dir, img_file)
            try:
                img_array = preprocess_image(img_path)
                embedding = embedding_model.predict(img_array, verbose=0)[0]
                embeddings.append(embedding)
                labels.append(label)
                image_paths.append(img_path)
            except Exception as e:
                print(f"  ⚠️  Error on {img_path}: {e}")

    embeddings  = np.array(embeddings)
    labels      = np.array(labels)
    image_paths = np.array(image_paths)

    np.save(os.path.join(PROCESSED_DIR, f"{split_name}_embeddings.npy"),  embeddings)
    np.save(os.path.join(PROCESSED_DIR, f"{split_name}_labels.npy"),      labels)
    np.save(os.path.join(PROCESSED_DIR, f"{split_name}_image_paths.npy"), image_paths)

    print(f"\n  ✅ {split_name.upper()} done")
    print(f"     Images processed : {len(embeddings)}")
    print(f"     Embedding shape  : {embeddings.shape}")
    print(f"     Saved to         : {PROCESSED_DIR}")

    return len(embeddings)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("  EMBEDDING EXTRACTION")
    print("=" * 60)

    if not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"\n❌ Model not found: {EMBEDDING_MODEL_PATH}")
        print("   Please run train_triplet.py first.")
        exit(1)

    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ Train dir not found: {TRAIN_DIR}")
        exit(1)

    if not os.path.exists(TEST_DIR):
        print(f"\n❌ Test dir not found: {TEST_DIR}")
        exit(1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"\nLoading model from: {EMBEDDING_MODEL_PATH}")
    embedding_model = load_embedding_model(EMBEDDING_MODEL_PATH)
    print(f"   Output shape: {embedding_model.output_shape}")

    train_count = process_split(embedding_model, TRAIN_DIR, "train")
    test_count  = process_split(embedding_model, TEST_DIR,  "test")

    print("\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    print(f"  Train embeddings : {train_count}")
    print(f"  Test  embeddings : {test_count}")
    print(f"  Saved to         : {PROCESSED_DIR}")
    print("=" * 60)