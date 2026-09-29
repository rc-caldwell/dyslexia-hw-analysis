import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report, roc_auc_score

# Must match exactly for weight loading
class ConstrainedAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.encoder_linear = nn.Linear(64 * 8 * 8, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, 64 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (64, 8, 8))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder_conv(self.unflatten(self.decoder_linear(self.encoder_linear(self.flatten(self.encoder_conv(x))))))

def load_test_paths(root_dir):
    normals, anomalies = [], []
    for p in Path(root_dir).rglob("*.*"):
        if p.suffix.lower() in ['.jpg', '.png', '.bmp'] and 'test' in [part.lower() for part in p.parts]:
            if 'reversal' in [part.lower() for part in p.parts] or 'corrected' in [part.lower() for part in p.parts]:
                anomalies.append(p)
            else:
                normals.append(p)
    return normals, anomalies

def build_mock_classroom(normals, anomalies, total_students=500, dyslexia_rate=0.15):
    students = []
    num_dyslexic = int(total_students * dyslexia_rate)
    
    # 1. Build Dyslexic Students (Mix of normal and 4-5 anomaly letters)
    for i in range(num_dyslexic):
        letters = random.sample(normals, 22) + random.sample(anomalies, 4)
        students.append({'student_id': f"Student_{i}_(Dyslexic)", 'true_status': 1, 'paths': letters})
        
    # 2. Build Neurotypical Students (All normal, but prone to standard sloppiness)
    for i in range(num_dyslexic, total_students):
        letters = random.sample(normals, 26)
        students.append({'student_id': f"Student_{i}_(Neurotypical)", 'true_status': 0, 'paths': letters})
        
    random.shuffle(students)
    return students

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConstrainedAutoencoder(latent_dim=32).to(device)
    model.load_state_dict(torch.load("data/processed/constrained_ae.pth", map_location=device, weights_only=True))
    model.eval()
    
    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((64, 64)), transforms.ToTensor()])
    criterion = nn.MSELoss(reduction='none')
    
    print("Loading test paths and generating mock classroom of 500 students...")
    normals, anomalies = load_test_paths("data/raw")
    classroom = build_mock_classroom(normals, anomalies)
    
    results = []
    print("Evaluating students via 1D Bottleneck MSE...")
    
    with torch.no_grad():
        for student in classroom:
            letter_mses = []
            for path in student['paths']:
                img = Image.open(path)
                tensor = transform(img).unsqueeze(0).to(device)
                recon = model(tensor)
                mse = criterion(recon, tensor).mean().item()
                letter_mses.append(mse)
            
            # The Screening Metric: Look at their worst handwriting mistakes (90th percentile)
            # A neurotypical student might have 1 sloppy letter, but a dyslexic student will have several structural failures
            risk_score = np.percentile(letter_mses, 90)
            
            results.append({
                'student_id': student['student_id'],
                'true_status': student['true_status'],
                'risk_score': risk_score
            })
            
    df = pd.DataFrame(results)
    roc = roc_auc_score(df['true_status'], df['risk_score'])
    print(f"\nStudent-Level ROC-AUC Score: {roc:.4f}")
    
    # In an unlabeled real-world scenario, you simply flag the top X percent of students with the highest risk scores.
    # We will flag the top 15% of the classroom for review.
    threshold = np.percentile(df['risk_score'], 85)
    df['flagged_for_review'] = (df['risk_score'] >= threshold).astype(int)
    
    print("\nClassroom Screening Results (0 = Normal, 1 = Dyslexic):")
    print(classification_report(df['true_status'], df['flagged_for_review']))