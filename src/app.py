import os
import numpy as np
from PIL import Image
import streamlit as st

# ----------------------------
# Paths
# ----------------------------
PROCESSED_DIR = "../data/processed"

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ----------------------------
# Load embeddings
# ----------------------------
@st.cache_resource
def load_data():
    embeddings = np.load(os.path.join(PROCESSED_DIR, "embeddings.npy"))
    labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy"))
    image_paths = np.load(os.path.join(PROCESSED_DIR, "image_paths.npy"))
    return embeddings, labels, image_paths

embeddings, labels, image_paths = load_data()

# ----------------------------
# Similarity functions
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def find_similar_images(query_idx, top_k=5):
    query_embedding = embeddings[query_idx]
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    top_k_indices = similarities.argsort()[-top_k-1:][::-1]
    top_k_indices = [idx for idx in top_k_indices if idx != query_idx][:top_k]
    return top_k_indices, similarities[top_k_indices]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Image Similarity Search", layout="wide")

st.title("🔍 AI Image Similarity Search (CIFAR-10)")
st.markdown("This demo uses learned embeddings to find visually similar images.")

top_k = st.slider("Number of similar images (K)", 1, 10, 5)

if st.button("🎲 Run Random Demo (like terminal)"):
    query_idx = np.random.randint(0, len(embeddings))

    top_k_indices, similarities = find_similar_images(query_idx, top_k)

    query_label = labels[query_idx]
    similar_labels = labels[top_k_indices]
    num_correct = int(np.sum(similar_labels == query_label))

    st.subheader("🖼️ Query Image")
    st.image(Image.open(image_paths[query_idx]), caption=f"Class: {CIFAR10_CLASSES[query_label]}", width=200)

    st.subheader(f"📊 Results — Accuracy: {num_correct}/{top_k}")

    cols = st.columns(top_k)

    for i, (idx, sim) in enumerate(zip(top_k_indices, similarities)):
        img = Image.open(image_paths[idx])
        label = labels[idx]

        is_correct = label == query_label
        status = "✅" if is_correct else "❌"

        with cols[i]:
            st.image(img, use_container_width=True)
            st.markdown(f"""
**Rank #{i+1}**  
Class: `{CIFAR10_CLASSES[label]}`  
Similarity: `{sim:.3f}`  
Match: {status}
""")

    st.success("Done! This is the same logic as your terminal demo.")
