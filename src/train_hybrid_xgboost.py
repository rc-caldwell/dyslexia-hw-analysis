import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import multiprocessing

# 1. Define Encoder Architecture
class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.encoder_linear = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x):
        return self.encoder_linear(self.flatten(self.encoder_conv(x)))

# 2. Batched Dataset Loader
class LatentExtractionDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []
        self.filenames = []
        self.splits = []
        self.labels = []
        
        for p in Path(root_dir).rglob("*.*"):
            if p.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            parts = [part.lower() for part in p.parts]
            split = 'train' if 'train' in parts else 'test' if 'test' in parts else None
            if not split:
                continue
                
            self.paths.append(p)
            self.filenames.append(p.name)
            self.splits.append(split)
            self.labels.append(1 if 'reversal' in parts or 'corrected' in parts else 0)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx])), self.filenames[idx], self.splits[idx], self.labels[idx]

def extract_and_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Encoder Weights
    encoder = Encoder(latent_dim=32).to(device)
    encoder.load_state_dict(torch.load("data/processed/constrained_ae.pth", map_location=device, weights_only=True), strict=False)
    encoder.eval()
    
    # Extract Embeddings in Batches
    print("Extracting deep latent embeddings...")
    dataset = LatentExtractionDataset("data/raw")
    cores = multiprocessing.cpu_count()
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=cores)
    
    latent_data = []
    with torch.no_grad():
        for imgs, filenames, splits, labels in dataloader:
            imgs = imgs.to(device)
            latents = encoder(imgs).cpu().numpy()
            
            for i in range(len(filenames)):
                row = {'filename': filenames[i], 'split': splits[i], 'label': labels[i].item()}
                for j in range(32):
                    row[f'latent_{j}'] = latents[i, j]
                latent_data.append(row)

    df_latent = pd.DataFrame(latent_data).drop_duplicates(subset=['filename'])
    
    print("Loading OpenCV geometric features and merging...")
    df_geo = pd.read_csv("data/processed/handwriting_features.csv")
    df_final = pd.merge(df_geo, df_latent, on='filename', how='inner')
    
    # Prepare Train/Test Sets
    train_df = df_final[df_final['split'] == 'train']
    test_df = df_final[df_final['split'] == 'test']
    
    feature_cols = ['center_x', 'center_y', 'aspect_ratio', 'solidity', 'extent', 'left_density', 'right_density'] + \
                   [f'latent_{i}' for i in range(32)]
    
    X_train, y_train = train_df[feature_cols], train_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']
    
    # Train XGBoost
    print("Training XGBoost Classifier...")
    # scale_pos_weight helps if the Reversal/Normal classes are imbalanced
    scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
    
    xgb = XGBClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    
    # Evaluate
    preds = xgb.predict(X_test)
    probs = xgb.predict_proba(X_test)[:, 1]
    
    print("\n--- Hybrid Model Performance ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Plot Feature Importance
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_importance(xgb, max_num_features=15, ax=ax, importance_type='gain', 
                    title="Top 15 Most Predictive Features (OpenCV vs Latent)")
    plt.tight_layout()
    plt.savefig("data/visualizations/xgboost_importance.png", dpi=300)
    print("\nFeature importance plot saved to data/visualizations/xgboost_importance.png")
    xgb.save_model("data/processed/hybrid_xgb.json")
    print("XGBoost model saved to data/processed/hybrid_xgb.json")
    

if __name__ == "__main__":
    extract_and_train()
