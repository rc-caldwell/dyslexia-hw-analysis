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
import base64

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
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.encoder_conv(x)
        if x.requires_grad: 
            x.register_hook(self.activations_hook)
        self.activations = x
        return self.encoder_linear(self.flatten(x))

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

    results = []        
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
        
        tensor.requires_grad = True
        encoder.zero_grad()
        
        latent_tensor = encoder(tensor)
        latent = latent_tensor.detach().cpu().numpy().flatten()

        # 3. XGBoost Inference
        vector = geo_features + latent.tolist()
        df_vector = pd.DataFrame([vector], columns=feature_cols)
        prob = xgb.predict_proba(df_vector)[0, 1]
        
        # 4. Generate Grad-CAM (Targeting the latent embedding activation)
        target = latent_tensor.norm()
        target.backward()
        
        gradients = encoder.gradients.cpu().data.numpy()[0]
        activations = encoder.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (64, 64))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        original_cv = np.array(pil_img.resize((64, 64)))
        original_bgr = cv2.cvtColor(original_cv, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.png', overlay)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

        student_probs.append(prob)
        results.append({
            "filename": file.filename,
            "probability": float(prob),
            "heatmap_b64": heatmap_b64
        })
        
        
    if not student_probs:
        return {"error": "Could not extract features from the provided images."}
        
    # Aggregate Risk Score (90th Percentile)
    risk_score = float(np.percentile(student_probs, 90))
    flagged = bool(risk_score >= 0.85) # Using the 85th percentile threshold from yesterday
    
    return {
        "status": "success",
        "images_processed": len(results),
        "aggregate_risk_score": round(risk_score, 4),
        "flagged_for_review": flagged,
        "results": results
    }
