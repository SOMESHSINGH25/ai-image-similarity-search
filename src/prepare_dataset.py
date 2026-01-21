import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

# Paths
RAW_DATA_DIR = "../data/raw/cifar10_images"
PROCESSED_DIR = "../data/processed"
EMBEDDING_MODEL_PATH = "../models/embedding_model.keras"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load the trained embedding model
print("Loading embedding model...")
model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, compile=False)
print("Model loaded successfully!")

# The model you trained IS the embedding model (128-dim output)
# No need to remove layers - it's already the base CNN
embedding_model = model

print(f"Embedding model output shape: {embedding_model.output_shape}")

# Image preprocessing function
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((64, 64))  # Your training used IMG_SIZE=64
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

embeddings = []
labels = []
image_paths = []

print("Processing images and creating embeddings...")
for label in sorted(os.listdir(RAW_DATA_DIR)):
    label_dir = os.path.join(RAW_DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    
    for img_file in tqdm(os.listdir(label_dir), desc=f"Label {label}"):
        img_path = os.path.join(label_dir, img_file)
        try:
            img_array = preprocess_image(img_path)
            embedding = embedding_model.predict(img_array, verbose=0)[0]
            
            embeddings.append(embedding)
            labels.append(int(label))
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)
image_paths = np.array(image_paths)

# Save embeddings, labels, and paths
np.save(os.path.join(PROCESSED_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(PROCESSED_DIR, "labels.npy"), labels)
np.save(os.path.join(PROCESSED_DIR, "image_paths.npy"), image_paths)

print(f"\n{'='*60}")
print(f"✓ Done! Processed {len(embeddings)} images.")
print(f"✓ Embedding dimension: {embeddings.shape[1]}")
print(f"✓ Embeddings saved to {PROCESSED_DIR}/embeddings.npy")
print(f"✓ Labels saved to {PROCESSED_DIR}/labels.npy")
print(f"✓ Image paths saved to {PROCESSED_DIR}/image_paths.npy")
print(f"{'='*60}")