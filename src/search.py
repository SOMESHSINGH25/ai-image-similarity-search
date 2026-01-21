import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import shutil

# ----------------------------
# Paths and parameters
# ----------------------------
RAW_DATA_DIR = "../data/raw/cifar10_images"
PROCESSED_DIR = "../data/processed"
EMBEDDING_MODEL_PATH = "../models/embedding_model.keras"
RESULTS_DIR = "../results"
TOP_K = 5  # Number of top similar images to retrieve

# Create results folder
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ----------------------------
# Load dataset embeddings
# ----------------------------
print("Loading processed embeddings, labels, and image paths...")
embeddings = np.load(os.path.join(PROCESSED_DIR, "embeddings.npy"))
labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy"))
image_paths = np.load(os.path.join(PROCESSED_DIR, "image_paths.npy"))
print(f"✓ Loaded {len(embeddings)} embeddings.")

# ----------------------------
# Compute cosine similarity
# ----------------------------
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def find_similar_images(query_idx, top_k=5):
    """Find top-k most similar images to the query"""
    query_embedding = embeddings[query_idx]
    
    # Compute similarities with all images
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    
    # Get top-k indices (excluding the query itself)
    top_k_indices = similarities.argsort()[-top_k-1:][::-1]
    top_k_indices = [idx for idx in top_k_indices if idx != query_idx][:top_k]
    
    return top_k_indices, similarities[top_k_indices]

# ----------------------------
# Visualization function
# ----------------------------
def visualize_results(query_idx, top_k_indices, similarities, save_path):
    """Create visualization of query and similar images"""
    fig, axes = plt.subplots(2, TOP_K + 1, figsize=(15, 6))
    
    # Query image
    query_img = Image.open(image_paths[query_idx])
    query_label = labels[query_idx]
    
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f'QUERY IMAGE\n{CIFAR10_CLASSES[query_label]}', 
                        fontsize=11, fontweight='bold', color='blue')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Similar images
    for i, (idx, sim) in enumerate(zip(top_k_indices, similarities)):
        img = Image.open(image_paths[idx])
        label = labels[idx]
        
        # Determine if match is correct (same class)
        is_correct = label == query_label
        color = 'green' if is_correct else 'red'
        
        axes[0, i+1].imshow(img)
        axes[0, i+1].set_title(
            f'Similar #{i+1}\n{CIFAR10_CLASSES[label]}\nScore: {sim:.3f}',
            fontsize=10, color=color
        )
        axes[0, i+1].axis('off')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")
    plt.close()

# ----------------------------
# Main search demo
# ----------------------------
def run_search_demo(num_queries=5):
    """Run similarity search demo with multiple queries"""
    print("\n" + "="*60)
    print("RUNNING SIMILARITY SEARCH DEMO")
    print("="*60)
    
    # Select random query images
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
    
    correct_matches = []
    
    for i, query_idx in enumerate(query_indices):
        print(f"\nQuery {i+1}/{num_queries}:")
        print(f"  Image: {image_paths[query_idx]}")
        print(f"  Class: {CIFAR10_CLASSES[labels[query_idx]]}")
        
        # Find similar images
        top_k_indices, similarities = find_similar_images(query_idx, TOP_K)
        
        # Count correct matches (same class)
        query_label = labels[query_idx]
        similar_labels = labels[top_k_indices]
        num_correct = np.sum(similar_labels == query_label)
        correct_matches.append(num_correct)
        
        print(f"  Similar images found:")
        for rank, (idx, sim) in enumerate(zip(top_k_indices, similarities), 1):
            match_status = "✓" if labels[idx] == query_label else "✗"
            print(f"    {rank}. {CIFAR10_CLASSES[labels[idx]]} (similarity: {sim:.3f}) {match_status}")
        
        print(f"  Accuracy: {num_correct}/{TOP_K} correct matches")
        
        # Create visualization
        save_path = os.path.join(RESULTS_DIR, f"similarity_search_{i+1}.png")
        visualize_results(query_idx, top_k_indices, similarities, save_path)
    
    # Overall statistics
    avg_correct = np.mean(correct_matches)
    accuracy = avg_correct / TOP_K * 100
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total queries: {num_queries}")
    print(f"Top-{TOP_K} accuracy: {accuracy:.1f}%")
    print(f"Average correct matches: {avg_correct:.1f}/{TOP_K}")
    print(f"Visualizations saved to: {RESULTS_DIR}/")
    print("="*60)

# ----------------------------
# Run the demo
# ----------------------------
if __name__ == "__main__":
    run_search_demo(num_queries=5)
    print("\n✓ DEMO COMPLETED! Check the 'results/' folder for visualizations.")