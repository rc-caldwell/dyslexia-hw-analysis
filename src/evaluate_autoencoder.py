import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import multiprocessing

# 1. Define the exact same architecture to load the weights
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

# 2. Create an Evaluation Dataset that returns labels and filenames
class EvalHandwritingDataset(Dataset):
    def __init__(self, root_dir, split='Test'):
        self.paths = []
        self.labels = [] # 0 for Normal, 1 for Anomaly
        self.filenames = []
        
        for p in Path(root_dir).rglob("*.*"):
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                parts_lower = [part.lower() for part in p.parts]
                if split.lower() in parts_lower:
                    self.paths.append(p)
                    self.filenames.append(p.name)
                    # Group Reversal and Corrected together as Anomalies (1)
                    if 'reversal' in parts_lower or 'corrected' in parts_lower:
                        self.labels.append(1)
                    else:
                        self.labels.append(0)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img), self.labels[idx], self.filenames[idx]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load("data/processed/autoencoder.pth", map_location=device, weights_only=True))
    model.eval() # Set to evaluation mode!
    
    # Load Test Data
    dataset = EvalHandwritingDataset("data/raw", split="test")
    cores = multiprocessing.cpu_count()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=cores)
    
    # reduction='none' allows us to get the loss per image instead of batch average
    criterion = nn.MSELoss(reduction='none') 
    
    all_losses = []
    all_labels = []
    all_filenames = []
    
    print(f"Evaluating {len(dataset)} test images...")
    
    with torch.no_grad(): # Disable gradient tracking for speed and memory
        for imgs, labels, filenames in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            
            # Calculate MSE per image: shape [batch_size, 1, 64, 64] -> [batch_size]
            loss = criterion(outputs, imgs)
            per_image_mse = loss.view(imgs.size(0), -1).mean(dim=1)
            
            all_losses.extend(per_image_mse.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)

    # 3. Analyze Results
    df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': all_labels,
        'reconstruction_loss': all_losses
    })
    
    # Calculate ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
    # This evaluates how well the loss separates the classes regardless of threshold
    roc_auc = roc_auc_score(df['true_label'], df['reconstruction_loss'])
    print(f"\nROC-AUC Score: {roc_auc:.4f} (1.0 is perfect, 0.5 is random guessing)")
    
    # Let's define an anomaly dynamically based on the dataset balance.
    # Since ~53% of the test set is Reversals/Corrected, we'll flag the top 53% highest losses.
    threshold = np.percentile(df['reconstruction_loss'], 47) 
    
    df['predicted_anomaly'] = (df['reconstruction_loss'] > threshold).astype(int)
    
    print(f"\n--- Model Performance using MSE Threshold: {threshold:.6f} ---")
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print(confusion_matrix(df['true_label'], df['predicted_anomaly']))
    
    print("\nClassification Report (0: Normal, 1: Anomaly):")
    print(classification_report(df['true_label'], df['predicted_anomaly']))
    
    # Save scored data
    df.to_csv("data/processed/autoencoder_eval_scores.csv", index=False)