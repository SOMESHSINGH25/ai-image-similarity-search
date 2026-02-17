import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PROCESSED_DIR = "../data/processed"
RESULTS_DIR   = "../results"
TOP_K         = 5
NUM_QUERIES   = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# =========================
# LOAD DATA
# =========================
print("Loading TRAIN embeddings...")
train_embeddings  = np.load(os.path.join(PROCESSED_DIR, "train_embeddings.npy"))
train_labels      = np.load(os.path.join(PROCESSED_DIR, "train_labels.npy"))
train_image_paths = np.load(os.path.join(PROCESSED_DIR, "train_image_paths.npy"))

print("Loading TEST embeddings...")
test_embeddings  = np.load(os.path.join(PROCESSED_DIR, "test_embeddings.npy"))
test_labels      = np.load(os.path.join(PROCESSED_DIR, "test_labels.npy"))
test_image_paths = np.load(os.path.join(PROCESSED_DIR, "test_image_paths.npy"))

print(f"✅ Train images : {len(train_embeddings)}")
print(f"✅ Test  images : {len(test_embeddings)}")

# =========================
# EUCLIDEAN DISTANCE
# Triplet loss is trained with
# Euclidean distance, so we
# must search with it too.
# We return similarity = 1/(1+dist)
# so higher = more similar (like before)
# =========================
def find_similar(query_embedding, top_k=TOP_K):
    # Euclidean distances to all train embeddings
    diffs     = train_embeddings - query_embedding   # (N, 128)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))  # (N,)

    # Closest = smallest distance = most similar
    top_k_indices = distances.argsort()[:top_k]

    # Convert to a 0-1 similarity score for display
    similarities = 1.0 / (1.0 + distances[top_k_indices])
    return top_k_indices, similarities

# =========================
# VISUALIZATION
# =========================
def visualize(query_idx, top_k_indices, similarities, save_path):
    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(15, 3))

    # Query image
    query_img   = Image.open(test_image_paths[query_idx])
    query_label = test_labels[query_idx]

    axes[0].imshow(query_img)
    axes[0].set_title(
        f"QUERY\n{CIFAR10_CLASSES[query_label]}",
        color="blue", fontweight="bold", fontsize=9
    )
    axes[0].axis("off")

    # Results
    for i, (idx, sim) in enumerate(zip(top_k_indices, similarities)):
        img        = Image.open(train_image_paths[idx])
        label      = train_labels[idx]
        is_correct = label == query_label
        color      = "green" if is_correct else "red"

        axes[i + 1].imshow(img)
        axes[i + 1].set_title(
            f"#{i+1} {CIFAR10_CLASSES[label]}\n{sim:.3f} {'✓' if is_correct else '✗'}",
            color=color, fontsize=9
        )
        axes[i + 1].axis("off")

    plt.suptitle(
        f"Query: {CIFAR10_CLASSES[query_label]} | Top-{TOP_K} Search Results",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# =========================
# MAIN EVALUATION
# =========================
def main():
    print("\n" + "=" * 60)
    print("  SIMILARITY SEARCH EVALUATION")
    print("=" * 60)

    if len(test_embeddings) < NUM_QUERIES:
        print(f"⚠️  Only {len(test_embeddings)} test images available.")
        indices = np.arange(len(test_embeddings))
    else:
        indices = np.random.choice(len(test_embeddings), NUM_QUERIES, replace=False)

    total_correct = 0

    for i, q_idx in enumerate(indices):
        print(f"\nQuery {i+1}/{NUM_QUERIES}")

        top_k_indices, sims = find_similar(test_embeddings[q_idx], TOP_K)

        query_label     = test_labels[q_idx]
        retrieved_labels = train_labels[top_k_indices]
        num_correct     = int(np.sum(retrieved_labels == query_label))
        total_correct  += num_correct

        print(f"  Query class    : {CIFAR10_CLASSES[query_label]}")
        print(f"  Correct Top-{TOP_K} : {num_correct}/{TOP_K}")

        for rank, (idx, sim) in enumerate(zip(top_k_indices, sims), 1):
            status = "✓" if train_labels[idx] == query_label else "✗"
            print(f"  {rank}. {CIFAR10_CLASSES[train_labels[idx]]:12s} sim={sim:.3f}  {status}")

        save_path = os.path.join(RESULTS_DIR, f"benchmark_{i+1}.png")
        visualize(q_idx, top_k_indices, sims, save_path)
        print(f"  Saved → {save_path}")

    accuracy = total_correct / (NUM_QUERIES * TOP_K) * 100

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Queries run      : {NUM_QUERIES}")
    print(f"  Top-{TOP_K} Accuracy : {accuracy:.2f}%")
    print(f"  Results saved to : {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()