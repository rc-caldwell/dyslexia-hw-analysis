This repository contains a machine learning pipeline for detecting structural anomalies in handwriting samples.

## Codebase Workflow

*   **Geometric Feature Extraction (OpenCV):** Calculates 7 spatial features per letter image (center x/y, aspect ratio, solidity, extent, left/right pixel density) to measure bounding box and symmetry imbalances.
*   **Deep Latent Extraction (PyTorch):** 
    *   Trains a Convolutional Autoencoder constrained by a 1D bottleneck (32 neurons).
    *   Extracts the 32-dimensional latent embedding for each image to capture structural variations and stroke textures that basic geometry misses.
*   **Hybrid Ensemble Classification (XGBoost):** 
    *   Merges the 7 OpenCV features and 32 PyTorch embeddings into a 39-dimensional feature vector.
    *   Trains an XGBoost classifier on this hybrid vector to flag individual anomalous characters.
*   **Subject-Level Screening Simulation:** 
    *   Groups isolated letters into unlabeled mock student profiles.
    *   Calculates a 90th-percentile risk score across a student's entire writing sample based on XGBoost probabilities.
    *   Flags high-risk profiles for manual review.

## Current Status & Next Steps
The core extraction scripts and model weights (`constrained_ae.pth`, `hybrid_xgb.json`) are finalized. 

Upcoming additions:
*   Wrap the OpenCV and PyTorch extraction steps into a single inference class.
*   Deploy the pipeline behind a **FastAPI** backend.
*   Build a **Streamlit** frontend for bulk image upload and risk score reporting.
EOF
