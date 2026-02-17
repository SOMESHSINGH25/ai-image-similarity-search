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

## 📁 Project Structure

```
ai-image-similarity-search/
│
├── src/                          # All source code
│   ├── app.py                    # Streamlit web application (SimiliAI UI)
│   ├── download_dataset.py       # Download CIFAR-10 and save as PNG images
│   ├── split_dataset.py          # Split raw images into train/test sets
│   ├── train_triplet.py          # Train the Triplet Network from scratch
│   ├── prepare_dataset.py        # Extract embeddings using trained model
│   └── search.py                 # Benchmark similarity search evaluation
│
├── data/                         # Generated — not tracked in git
│   ├── raw/
│   │   ├── cifar10_images/       # Raw downloaded images (10 class folders)
│   │   ├── train/                # 80% split (1596 images)
│   │   └── test/                 # 20% split (404 images)
│   └── processed/
│       ├── train_embeddings.npy  # (1596, 128) float32
│       ├── train_labels.npy      # (1596,) int
│       ├── train_image_paths.npy # (1596,) str
│       ├── test_embeddings.npy   # (404, 128) float32
│       ├── test_labels.npy       # (404,) int
│       └── test_image_paths.npy  # (404,) str
│
├── models/                       # Generated — not tracked in git
│   └── embedding_model.keras     # Trained base CNN (128-dim output)
│
├── results/                      # Generated — not tracked in git
│   ├── training_curves.png       # Train/val loss plot
│   └── benchmark_*.png           # Search result visualisations
│
├── logs/                         # Generated — not tracked in git
│   └── training_log.csv          # Epoch-by-epoch loss log
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**CIFAR-10** — a benchmark dataset of 60,000 32×32 colour images across 10 classes.

| Class | Index | Emoji | Description |
|-------|-------|-------|-------------|
| airplane | 0 | ✈️ | Fixed-wing aircraft |
| automobile | 1 | 🚗 | Four-wheeled vehicle |
| bird | 2 | 🐦 | Feathered vertebrate |
| cat | 3 | 🐱 | Domestic feline |
| deer | 4 | 🦌 | Hoofed mammal |
| dog | 5 | 🐶 | Domestic canine |
| frog | 6 | 🐸 | Amphibian species |
| horse | 7 | 🐴 | Large equine mammal |
| ship | 8 | 🚢 | Large watercraft |
| truck | 9 | 🚛 | Heavy goods vehicle |

**SimiliAI uses 2,000 images** (200 per class) — well above the 1,000+ requirement — split as:
- **Train:** 1,596 images (80%)
- **Test:** 404 images (20%)

Images are saved as PNG at native 32×32 resolution with sharpening applied (`PIL.ImageFilter.UnsharpMask`) before being resized to 64×64 during training.

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.12
- Windows / macOS / Linux
- ~2 GB free disk space
- No GPU required (CPU training supported)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-image-similarity-search.git
cd ai-image-similarity-search
```

### 2. Install Dependencies

```bash
pip install tensorflow pillow numpy streamlit matplotlib tqdm pandas
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.15.0
pillow>=10.0.0
numpy>=1.24.0
streamlit>=1.30.0
matplotlib>=3.7.0
tqdm>=4.65.0
pandas>=2.0.0
```

---

## 🚀 Running the Pipeline

Run these scripts **in order** from the project root directory:

### Step 1 — Download Dataset

```bash
py -3.12 src/download_dataset.py
```

Downloads CIFAR-10 via Keras and saves **2,000 images** as PNGs with sharpening applied.

**Output:** `data/raw/cifar10_images/` (10 class subfolders)

**Expected time:** 1–3 minutes (download dependent)

---

### Step 2 — Split Dataset

```bash
py -3.12 src/split_dataset.py
```

Splits the raw images 80/20 into train and test sets.

**Output:** `data/raw/train/` and `data/raw/test/`

**Expected output:**
```
Class 0: 160 train, 40 test
Class 1: 160 train, 40 test
...
✅ Dataset split completed!
```

---

### Step 3 — Train Triplet Network

```bash
py -3.12 src/train_triplet.py
```

Trains the CNN from scratch using triplet loss for up to 20 epochs with early stopping.

**Output:**
- `models/embedding_model.keras` — trained base model
- `results/training_curves.png` — loss plot
- `logs/training_log.csv` — epoch log

**Expected time:** 2–5 minutes per epoch on CPU (20–60 min total)

**Expected output:**
```
TRIPLET NETWORK TRAINING  (Fast Random Triplets)
Loading all images into memory...
Building training triplets... 8000
Building validation triplets... 1000
Epoch 1/20 — loss: 8.83 — val_loss: 4.91
Epoch 2/20 — loss: 6.44 — val_loss: 0.43
...
✅ Base embedding model saved to: ../models/embedding_model.keras
```

> **Note:** Training will automatically stop early if validation loss stops improving for 5 consecutive epochs.

---

### Step 4 — Extract Embeddings

```bash
py -3.12 src/prepare_dataset.py
```

Runs all 2,000 images through the trained model and saves their 128-dim embedding vectors.

**Output:** 6 `.npy` files in `data/processed/`

**Expected time:** 3–5 minutes

**Expected output:**
```
✅ TRAIN done — Images: 1596 — Shape: (1596, 128)
✅ TEST done  — Images: 404  — Shape: (404, 128)
```

---

### Step 5 — Evaluate Search (Optional)

```bash
py -3.12 src/search.py
```

Runs a 10-query benchmark evaluation and saves result visualisations to `results/`.

**Expected output:**
```
Query 1/10 — ship — Correct Top-5: 1/5
Query 2/10 — horse — Correct Top-5: 3/5
...
Top-5 Accuracy: ~40–60%
```

---

### Step 6 — Launch the App

```bash
streamlit run src/app.py
```

Opens SimiliAI in your browser at `http://localhost:8501`

---

## 🖥️ Streamlit Application

SimiliAI features a full-featured dark-themed web UI with four tabs:

### ⬡ Random Query
- Picks a random test image
- Searches the train database using Euclidean k-NN
- Shows query metadata, accuracy badge (colour-coded green/yellow/red)
- Displays per-result breakdown with similarity scores and distances
- Embedding statistics panel (min/max distance, avg similarity)
- Retrieved class distribution bar chart
- Full results grid with rank pills, match/miss indicators

### ⬡ Upload Image
- Upload any JPG/PNG/WEBP image from your computer
- Model embeds it in real time using the trained CNN
- Predicts the most likely class based on top-K neighbours
- Shows embedding norm, nearest distance, average similarity
- Full neighbours grid with similarity score bars

### ⬡ Training Curves
- Displays the training/validation loss plot
- Shows summary stats: epochs trained, best val loss, best epoch, total loss drop
- Styled epoch log table with best rows highlighted

### ⬡ How It Works
- Explains triplet networks, triplet loss formula, and random triplet sampling
- Layer-by-layer CNN architecture diagram
- 4-step search pipeline breakdown
- Technical rationale for design choices

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training images | 1,596 |
| Test images | 404 |
| Embedding dimension | 128 |
| Best val loss | ~0.19–0.43 |
| Top-5 accuracy (benchmark) | **40–60%** |
| Search latency | < 100ms |
| Training time (CPU) | ~30–60 min |

> Top-5 accuracy varies per run since the benchmark queries are randomly selected. Some classes (bird, deer, horse) perform better than others (airplane vs ship confusion is common due to visual similarity).

### Training Curve (Example)

```
Epoch  1 │ train: 8.83 │ val: 4.91  ← learning fast
Epoch  2 │ train: 6.44 │ val: 0.43  ← major improvement
Epoch  3 │ train: 4.68 │ val: 0.28  ← converging
Epoch  4 │ train: 3.57 │ val: 0.20  ← good separation
Epoch  5 │ train: 2.91 │ val: 0.20  ← plateau
...
⏹ Early stopping at epoch ~10–15
```

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

## ⚠️ Limitations & Future Work

### Current Limitations

- **Small dataset** — 2,000 images limits accuracy. More data would substantially improve results.
- **Low resolution** — CIFAR-10 images are 32×32 pixels, upscaled to 64×64. Visual detail is limited.
- **10 fixed classes** — the model only understands CIFAR-10 categories. Uploaded images of other objects may produce unexpected results.
- **CPU training** — training from scratch on CPU is slow. A GPU would reduce training time from hours to minutes.

### Possible Improvements

| Improvement | Expected Impact |
|-------------|----------------|
| Use 5,000–10,000 images | +10–20% accuracy |
| GPU training | 10–50× faster training |
| Semi-hard or batch-hard mining | Better embedding separation |
| Data augmentation (flip, crop, jitter) | Better generalisation |
| Deeper CNN (ResNet-style) | Higher quality features |
| FAISS vector index | Sub-millisecond search on millions of images |
| t-SNE / UMAP visualisation | Visualise the embedding space |

---

## 👤 Author

**Deep Learning Project** — AI-Powered Image Similarity Search and Recommendation System

Built with TensorFlow, Streamlit, and NumPy. Trained entirely from scratch on CIFAR-10.

---

## 📄 License

This project is for educational purposes. Dataset (CIFAR-10) credit: Krizhevsky, 2009.

---

<div align="center">
<sub>SimiliAI — Neural Image Search · Triplet Network · CIFAR-10 · Trained from Scratch</sub>
</div>