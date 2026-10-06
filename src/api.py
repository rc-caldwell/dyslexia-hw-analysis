from fastapi import FastAPI, UploadFile, File, Form
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
import re
import string

app = FastAPI(title="Dysgraphia Screening API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the 36 possible characters (a-z, 0-9) for target-aware routing
TARGET_CHARS = list(string.ascii_lowercase + string.digits)

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

class LatentGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_latent_idx=27):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, target_latent_idx]
        target.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        
        cam -= torch.min(cam)
        if torch.max(cam) != 0:
            cam /= torch.max(cam)
        return cam.cpu().numpy(), output.detach().cpu().numpy().flatten()

encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("data/processed/constrained_ae.pth", map_location=device, weights_only=True), strict=False)
encoder.eval()

grad_cam = LatentGradCAM(encoder, encoder.encoder_conv[4])

xgb = XGBClassifier()
xgb.load_model("data/processed/hybrid_xgb.json")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def extract_geometry_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None, []
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area, rect_area = cv2.contourArea(c), w * h
    
    if rect_area == 0 or h == 0: return None, []

    center_x, center_y = x + w / 2, y + h / 2
    aspect_ratio = float(w) / h
    hull_area = cv2.contourArea(cv2.convexHull(c))
    
    solidity = float(area) / hull_area if hull_area > 0 else 0
    extent = float(area) / rect_area
    
    left_density = cv2.countNonZero(thresh[y:y+h, x:x + w//2]) / (rect_area / 2)
    right_density = cv2.countNonZero(thresh[y:y+h, x + w//2:x+w]) / (rect_area / 2)
    
    features = [center_x, center_y, aspect_ratio, solidity, extent, left_density, right_density]
    return features

@app.post("/screen")
async def screen_student_sample(
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.85) 
):
    target_cols = [f'target_{c}' for c in TARGET_CHARS]
    feature_cols = ['center_x', 'center_y', 'aspect_ratio', 'solidity', 'extent', 'left_density', 'right_density'] + \
                   [f'latent_{i}' for i in range(32)] + \
                   target_cols
                   
    student_probs = []
    letter_details = []
    processed_count = 0
    
    for file in files:
        image_bytes = await file.read()
        
        geo_features = extract_geometry_from_bytes(image_bytes)
        if not geo_features: continue
            
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
        tensor = transform(pil_img).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        
        cam, latent = grad_cam.generate(tensor, target_latent_idx=27)
        
        # Extract intended character from filename and one-hot encode
        encoding = {f'target_{c}': 0 for c in TARGET_CHARS}
        match = re.match(r'^([a-zA-Z0-9]+)[-_.]', file.filename)
        if match:
            char = match.group(1).lower()
            if f'target_{char}' in encoding:
                encoding[f'target_{char}'] = 1
        one_hot_vector = list(encoding.values())
        
        # Concatenate into the final 75-length array
        vector = geo_features + latent.tolist() + one_hot_vector
        
        prob = float(xgb.predict_proba(pd.DataFrame([vector], columns=feature_cols))[0, 1])
        student_probs.append(prob)
        
        cam_resized = cv2.resize(cam, (pil_img.width, pil_img.height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        original_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)
        superimposed = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)
        
        _, buffer = cv2.imencode('.png', superimposed)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        letter_details.append({
            "filename": file.filename,
            "probability": prob,
            "heatmap_b64": b64_img,
        })
        processed_count += 1
        
    if not student_probs:
        return {"error": "Could not extract features from the provided images."}
        
    risk_score = float(np.percentile(student_probs, 90))
    
    return {
        "status": "success",
        "images_processed": processed_count,
        "aggregate_risk_score": round(risk_score, 4),
        "threshold_used": threshold,
        "flagged_for_review": bool(risk_score >= threshold),
        "letter_details": letter_details
    }