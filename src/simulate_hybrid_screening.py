import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import random
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from extract_features import extract_geometric_features # Reusing your OpenCV script

# 1. Define Encoder
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

def build_mock_classroom(root_dir, total_students=500, dyslexia_rate=0.15):
    normals, anomalies = [], []
    for p in Path(root_dir).rglob("*.*"):
        if p.suffix.lower() in ['.jpg', '.png'] and 'test' in [part.lower() for part in p.parts]:
            if 'reversal' in [part.lower() for part in p.parts] or 'corrected' in [part.lower() for part in p.parts]:
                anomalies.append(p)
            else:
                normals.append(p)
                
    students = []
    num_dyslexic = int(total_students * dyslexia_rate)
    
    for i in range(num_dyslexic):
        letters = random.sample(normals, 22) + random.sample(anomalies, 4)
        students.append({'student_id': f"Student_{i}_(Dyslexic)", 'true_status': 1, 'paths': letters})
        
    for i in range(num_dyslexic, total_students):
        letters = random.sample(normals, 26)
        students.append({'student_id': f"Student_{i}_(Neurotypical)", 'true_status': 0, 'paths': letters})
        
    random.shuffle(students)
    return students

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PyTorch Encoder
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("data/processed/constrained_ae.pth", map_location=device, weights_only=True), strict=False)
    encoder.eval()
    
    # Load Trained XGBoost Model
    xgb = XGBClassifier()
    xgb.load_model("data/processed/hybrid_xgb.json") # Ensure you add xgb.save_model("data/processed/hybrid_xgb.json") to your training script!
    
    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((64, 64)), transforms.ToTensor()])
    
    print("Generating unlabeled classroom of 500 students...")
    classroom = build_mock_classroom("data/raw")
    
    feature_cols = ['center_x', 'center_y', 'aspect_ratio', 'solidity', 'extent', 'left_density', 'right_density'] + \
                   [f'latent_{i}' for i in range(32)]
                   
    results = []
    print("Evaluating students via Hybrid XGBoost probabilities...")
    
    for student in classroom:
        student_probs = []
        for path in student['paths']:
            # 1. Get OpenCV Features
            geo_features = extract_geometric_features(path)
            if not geo_features: continue
            
            # 2. Get Latent Features
            img = Image.open(path)
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = encoder(tensor).cpu().numpy().flatten()
                
            # 3. Combine and Predict
            row = [geo_features[col] for col in feature_cols[:7]] + latent.tolist()
            prob = xgb.predict_proba(pd.DataFrame([row], columns=feature_cols))[0, 1]
            student_probs.append(prob)
            
        # The Screening Metric: Aggregate the student's highest risk probabilities
        risk_score = np.percentile(student_probs, 90)
        
        results.append({
            'student_id': student['student_id'],
            'true_status': student['true_status'],
            'risk_score': risk_score
        })
        
    df = pd.DataFrame(results)
    roc = roc_auc_score(df['true_status'], df['risk_score'])
    print(f"\nStudent-Level ROC-AUC Score: {roc:.4f}")
    
    # Flag top 15% of classroom
    threshold = np.percentile(df['risk_score'], 85)
    df['flagged_for_review'] = (df['risk_score'] >= threshold).astype(int)
    
    print("\nClassroom Screening Results (0 = Normal, 1 = Dyslexic):")
    print(classification_report(df['true_status'], df['flagged_for_review']))