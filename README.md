# AI-Powered Image Similarity Search & Recommendation System

Deep Learning Project using Triplet Networks for Visual Similarity Search

---

## Project Overview

This project builds an AI-powered image similarity search system that finds visually similar images using deep learning instead of text tags.

The system uses a Triplet Network to learn an embedding space where:
- Similar images are closer
- Different images are far apart

---

## Objectives

- Learn visual features using deep learning
- Generate embeddings for images
- Perform similarity search using cosine similarity
- Return Top-K most similar images
- Work without text labels
- Support 1000+ images

---

## Model

- CNN-based embedding network
- Input size: 64×64 RGB
- Output: 128-dimensional embedding vector
- Loss: Triplet Loss

---

## Dataset

- Based on CIFAR-10
- Total images used: 2000
- 10 classes:
  airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

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

## How To Run

1. Train the model:

python src/train_triplet.py

markdown
Copy code

2. Prepare embeddings:

python src/prepare_dataset.py

markdown
Copy code

3. Run similarity search:

python src/search.py

yaml
Copy code

Results will be saved in:

results/

yaml
Copy code

---

## Current Status

- Working similarity search system
- 2000 images indexed
- Top-5 accuracy ~36%
- Embedding size: 128

---

## Output

For each run, the system:
- Picks random query images
- Finds Top-5 similar images
- Saves visualization images in results folder

---

## Conclusion

This is a working AI-based image similarity search engine using deep learning and Triplet Loss.
