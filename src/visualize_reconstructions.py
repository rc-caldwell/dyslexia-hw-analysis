import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def visualize_reconstructions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load("data/processed/autoencoder.pth", map_location=device, weights_only=True))
    model.eval()
    
    # Load Scores to find extreme examples
    df = pd.read_csv("data/processed/autoencoder_eval_scores.csv")
    
    # Find 3 Top Anomalies (True Positives) and 3 Most Normal (True Negatives)
    true_positives = df[(df['true_label'] == 1) & (df['predicted_anomaly'] == 1)].sort_values('reconstruction_loss', ascending=False).head(3)
    true_negatives = df[(df['true_label'] == 0) & (df['predicted_anomaly'] == 0)].sort_values('reconstruction_loss', ascending=True).head(3)
    
    sample_files = pd.concat([true_positives, true_negatives])['filename'].tolist()
    labels = ["Anomaly (True Positive)"] * 3 + ["Normal (True Negative)"] * 3
    
    # Locate files
    img_paths = []
    for f in sample_files:
        found = list(Path("data/raw").rglob(f))
        if found:
            img_paths.append(found[0])
            
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    fig, axes = plt.subplots(len(img_paths), 3, figsize=(10, 3 * len(img_paths)))
    fig.suptitle("Original vs. Autoencoder Reconstruction vs. Error Heatmap", fontsize=16)
    
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for i, img_path in enumerate(img_paths):
            # Prepare image
            img = Image.open(img_path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Reconstruct
            output = model(img_tensor)
            
            # Calculate pixel-wise error
            error_map = criterion(output, img_tensor).squeeze().cpu().numpy()
            
            # Convert tensors for plotting
            orig_img = img_tensor.squeeze().cpu().numpy()
            recon_img = output.squeeze().cpu().numpy()
            
            # Plot Original
            axes[i, 0].imshow(orig_img, cmap='gray')
            axes[i, 0].set_title(f"Original\n{labels[i]}")
            axes[i, 0].axis('off')
            
            # Plot Reconstruction
            axes[i, 1].imshow(recon_img, cmap='gray')
            axes[i, 1].set_title("Reconstruction")
            axes[i, 1].axis('off')
            
            # Plot Error Map (Heatmap)
            # Warmer colors mean the autoencoder failed heavily on those specific pixels
            im = axes[i, 2].imshow(error_map, cmap='hot', interpolation='nearest')
            axes[i, 2].set_title("MSE Error Heatmap")
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    plt.savefig("data/visualizations/reconstruction_analysis.png", dpi=300)
    print("Saved visualization to data/visualizations/reconstruction_analysis.png")

if __name__ == "__main__":
    visualize_reconstructions()