from fastapi import FastAPI, UploadFile, File
from typing import List
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import io
import cv2

# 1. Initialize App and Models
app = FastAPI(title="Dysgraphia Screening API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load models globally so they stay in memory between API calls
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("data/processed/constrained_ae.pth", map_location=device, weights_only=True), strict=False)
encoder.eval()

xgb = XGBClassifier()
xgb.load_model("data/processed/hybrid_xgb.json")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 2. In-Memory OpenCV Extractor
def extract_geometry_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    rect_area = w * h
    
    if rect_area == 0 or h == 0:
        return None

    center_x = x + w / 2
    center_y = y + h / 2
    aspect_ratio = float(w) / h
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    extent = float(area) / rect_area
    
    left_half = thresh[y:y+h, x:x + w//2]
    right_half = thresh[y:y+h, x + w//2:x+w]
    left_density = cv2.countNonZero(left_half) / (rect_area / 2)
    right_density = cv2.countNonZero(right_half) / (rect_area / 2)
    
    return [center_x, center_y, aspect_ratio, solidity, extent, left_density, right_density]

# 3. The API Endpoint
@app.post("/screen")
async def screen_student_sample(files: List[UploadFile] = File(...)):
    feature_cols = ['center_x', 'center_y', 'aspect_ratio', 'solidity', 'extent', 'left_density', 'right_density'] + \
                   [f'latent_{i}' for i in range(32)]
                   
    student_probs = []
    processed_count = 0
    
    for file in files:
        image_bytes = await file.read()
        
        # 1. OpenCV Features
        geo_features = extract_geometry_from_bytes(image_bytes)
        if not geo_features:
            continue
            
        # 2. PyTorch Latent Features
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
        tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent = encoder(tensor).cpu().numpy().flatten()
            
        # 3. XGBoost Inference
        vector = geo_features + latent.tolist()
        df_vector = pd.DataFrame([vector], columns=feature_cols)
        prob = xgb.predict_proba(df_vector)[0, 1]
        
        student_probs.append(prob)
        processed_count += 1
        
    if not student_probs:
        return {"error": "Could not extract features from the provided images."}
        
    # Aggregate Risk Score (90th Percentile)
    risk_score = float(np.percentile(student_probs, 90))
    flagged = bool(risk_score >= 0.85) # Using the 85th percentile threshold from yesterday
    
    return {
        "status": "success",
        "images_processed": processed_count,
        "aggregate_risk_score": round(risk_score, 4),
        "flagged_for_review": flagged,
        "letter_breakdown": [round(float(p), 4) for p in student_probs]
    }
