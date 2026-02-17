# SimiliAI — Neural Image Similarity Search

<div align="center">

```
 ███████╗██╗███╗   ███╗██╗██╗      ██╗ █████╗ ██╗
 ██╔════╝██║████╗ ████║██║██║      ██║██╔══██╗██║
 ███████╗██║██╔████╔██║██║██║      ██║███████║██║
 ╚════██║██║██║╚██╔╝██║██║██║      ██║██╔══██║██║
 ███████║██║██║ ╚═╝ ██║██║███████╗ ██║██║  ██║██║
 ╚══════╝╚═╝╚═╝     ╚═╝╚═╝╚══════╝ ╚═╝╚═╝  ╚═╝╚═╝
```

**AI-Powered Image Similarity Search using Triplet Networks**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

*Find visually similar images in milliseconds using deep learned embeddings — trained entirely from scratch.*

</div>

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Triplet Loss](#-triplet-loss)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Setup & Installation](#-setup--installation)
- [Running the Pipeline](#-running-the-pipeline)
- [Streamlit Application](#-streamlit-application)
- [Results](#-results)
- [Scripts Reference](#-scripts-reference)
- [Technical Details](#-technical-details)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🧠 Project Overview

**SimiliAI** is a deep learning system that learns to identify and retrieve visually similar images from a large database — without using any text labels, manual tags, or pre-trained models. Everything is **trained from scratch**.

Given a query image, SimiliAI:
1. Encodes it into a compact **128-dimensional embedding vector**
2. Searches a pre-built database of embeddings using **Euclidean distance**
3. Returns the **top-K most visually similar images** in real time

This mirrors real-world systems like Pinterest's visual search, Getty Images' visual discovery, e-commerce "shop the look" features, and Google Lens.

---

## ❓ Problem Statement

In today's digital world, we are surrounded by millions of images across social media, e-commerce, photo galleries, and digital libraries. Users often struggle to find visually similar images or get relevant recommendations based on image content.

**Traditional text-based search fails** to capture visual semantics — a photo of a golden retriever and a labrador look identical to a human but have completely different tags.

**Challenge:** How can we automatically identify and recommend visually similar images from a large database without relying on manual tags or text descriptions?

**Real-world applications:**
- 🛒 **E-commerce** — "Find similar products" on shopping websites
- 📸 **Social media** — Instagram's "Related Posts" recommendations
- 🖼️ **Stock photography** — Getty Images' visual search
- 👗 **Fashion** — "Shop the Look" features

---

## ⚙️ How It Works

SimiliAI uses a **Triplet Network** — a Siamese-style architecture that learns a metric embedding space where:

```
distance(same class images)     →  SMALL
distance(different class images) →  LARGE
```

### Training Phase
The model trains on **triplets** of images:

```
Anchor  (A) ──┐
Positive (P) ─┼──► CNN ──► Embeddings ──► Triplet Loss
Negative (N) ─┘
```

- **Anchor** — a reference image
- **Positive** — a different image of the **same class** as anchor
- **Negative** — an image of a **different class** from anchor

The loss function teaches the network: *"keep anchor close to positive, push anchor away from negative."*

### Search Phase
At query time:
1. Query image → CNN → 128-dim embedding vector
2. Compute Euclidean distance from query to all database embeddings
3. Sort by distance (smallest = most similar)
4. Return top-K results

---

## 🏗️ Architecture

### Base CNN (Embedding Model)

```
Input (64×64×3)
       │
  ┌────▼────┐
  │ Conv2D  │  32 filters, 3×3, ReLU, Same padding
  │ BatchNorm│
  │ MaxPool │  2×2
  └────┬────┘
       │
  ┌────▼────┐
  │ Conv2D  │  64 filters, 3×3, ReLU, Same padding
  │ BatchNorm│
  │ MaxPool │  2×2
  └────┬────┘
       │
  ┌────▼────┐
  │ Conv2D  │  128 filters, 3×3, ReLU, Same padding
  │ BatchNorm│
  │ GlobalAvgPool│
  └────┬────┘
       │
  ┌────▼────┐
  │ Dense   │  256 units, ReLU
  │ Dropout │  0.3
  │ Dense   │  128 units, tanh
  └────┬────┘
       │
  128-dim Embedding ∈ [-1, 1]
```

**Total parameters:** ~2.2M (8.49 MB)

### Training Model (Triplet Wrapper)

The training model wraps the base CNN three times (shared weights):

```
Anchor  ──► [Base CNN] ──┐
Positive ──► [Base CNN] ──┼──► Concatenate ──► Triplet Loss
Negative ──► [Base CNN] ──┘
```

After training, only the **Base CNN** is saved and used for inference.

### Key Design Choices

| Choice | Reason |
|--------|--------|
| `tanh` output activation | Bounds embeddings to [-1, 1], prevents embedding collapse |
| `GlobalAveragePooling2D` | Better than Flatten for small datasets — reduces overfitting |
| `BatchNormalization` | Stabilises training, allows higher learning rates |
| `Dropout(0.3)` | Regularisation for 2000-image dataset |
| Euclidean distance search | Matches the triplet loss metric — correct at inference time |

---

## 📐 Triplet Loss

The core learning signal:

```
L(A, P, N) = max( d(A, P) - d(A, N) + α, 0 )
```

Where:
- `d(x, y) = √Σ(xᵢ - yᵢ)²` — Euclidean distance
- `A` — Anchor embedding
- `P` — Positive embedding (same class)
- `N` — Negative embedding (different class)
- `α = 0.3` — Margin (minimum separation required)

**Intuition:**
- If `d(A,P) + α < d(A,N)` — loss = 0 (already well separated ✓)
- If `d(A,N) - d(A,P) < α` — positive loss (needs more separation ✗)

### Triplet Sampling Strategy

SimiliAI uses **pre-built random triplet sampling**:
- 8,000 training triplets built upfront in RAM (pure NumPy)
- 1,000 validation triplets
- Passed to `model.fit()` in batches of 32
- No per-step model calls during data generation → **10× faster** than online mining

This approach is optimal for CPU training on small datasets.

---
```
## Folder Structure

AI_Image_Similarity_Search/
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ └── embedding_model.keras
├── results/
├── src/
│ ├── train_triplet.py
│ ├── prepare_dataset.py
│ └── search.py
├── README.md
└── requirements.txt

---
```
## How To Run

1. Train the model:

python src/train_triplet.py


2. Prepare embeddings:

python src/prepare_dataset.py


3. Run similarity search:

python src/search.py


Results will be saved in:

results/


---

## Current Status

- Working similarity search system
- 2000 images indexed
- Top-5 accuracy ~36%
- Embedding size: 128

---

## 📜 Scripts Reference

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `download_dataset.py` | Download CIFAR-10, save as PNG with sharpening | — | `data/raw/cifar10_images/` |
| `split_dataset.py` | 80/20 train/test split | `cifar10_images/` | `data/raw/train/`, `data/raw/test/` |
| `train_triplet.py` | Train CNN with triplet loss | `data/raw/train/` | `models/embedding_model.keras`, `results/training_curves.png`, `logs/training_log.csv` |
| `prepare_dataset.py` | Extract 128-dim embeddings | `models/embedding_model.keras` | `data/processed/*.npy` (6 files) |
| `search.py` | Benchmark k-NN search evaluation | `data/processed/*.npy` | `results/benchmark_*.png`, console report |
| `app.py` | Streamlit web application | `data/processed/*.npy`, `models/embedding_model.keras` | Interactive UI at localhost:8501 |

---

## 🔬 Technical Details

### Why Euclidean Distance (not Cosine Similarity)?

The triplet loss is defined as:
```
L = max( d(A,P) - d(A,N) + α, 0 )
```
where `d` is **Euclidean distance**. This means the embedding space is geometrically shaped by Euclidean relationships during training. Using cosine similarity at search time would measure angles instead of distances — ignoring the actual structure learned by the loss function and producing incorrect rankings.

### Why `tanh` on the Output Layer?

Without output normalisation, CNN embeddings can collapse — all vectors pointing in nearly the same direction, making cosine similarity always ≈ 1.0 and Euclidean distances meaningless. `tanh` bounds each embedding dimension to `[-1, 1]`, which:
- Prevents embedding collapse
- Keeps distances in a meaningful range
- Does not enforce equal magnitude (unlike L2 normalisation, which caused degenerate results)

### Why GlobalAveragePooling2D (not Flatten)?

On small datasets like 2,000 images, `Flatten` produces a huge feature vector (8,192-dim) that leads to severe overfitting. `GlobalAveragePooling2D` averages each feature map to a single value, producing a 128-dim vector that captures spatial patterns without memorising positions — much better generalisation.

### Why Random Triplets (not Semi-Hard Mining)?

Semi-hard mining calls the model during data generation for every single triplet, which on CPU means 10–40 minutes per epoch. By pre-building 8,000 triplets in RAM using pure NumPy before training begins and using `model.fit()` for batching, each epoch completes in 2–5 minutes — a 10× speedup with comparable accuracy on this dataset size.

---

## Conclusion

This is a working AI-based image similarity search engine using deep learning and Triplet Loss.
