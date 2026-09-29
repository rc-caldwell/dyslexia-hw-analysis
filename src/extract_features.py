import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import concurrent.futures
from tqdm import tqdm
import multiprocessing

def extract_geometric_features(img_path):
    """Extracts spatial and geometric metrics from a single letter image."""
    # Read as grayscale and binarize (ink becomes white 255, background black 0)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    features = {'filename': img_path.name}
    
    # 1. Center of Mass (Spatial Asymmetry)
    # Detects if a letter is heavier on the left/right, catching partial reversals
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        features['center_x'] = M["m10"] / M["m00"]
        features['center_y'] = M["m01"] / M["m00"]
    else:
        features['center_x'], features['center_y'] = 0, 0
        
    # 2. Contours and Bounding Box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest continuous ink block is the main letter body
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Aspect Ratio (Width / Height)
        # Identifies squashed or overly stretched characters
        x, y, w, h = cv2.boundingRect(c)
        features['aspect_ratio'] = w / float(h) if h > 0 else 0
        
        # Solidity (Letter Area / Convex Hull Area)
        # A low solidity indicates broken loops, jagged edges, or heavy tremors
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = float(area) / hull_area if hull_area > 0 else 0
        
        # Extent (Letter Area / Bounding Box Area)
        # Captures how much the letter fills its allotted space
        rect_area = w * h
        features['extent'] = float(area) / rect_area if rect_area > 0 else 0
        
        # Left vs Right Density (Reversal Detection)
        # Compares pixel mass on the left vs right side of the bounding box centerline
        half_w = w // 2
        left_side = thresh[y:y+h, x:x+half_w]
        right_side = thresh[y:y+h, x+half_w:x+w]
        features['left_density'] = cv2.countNonZero(left_side)
        features['right_density'] = cv2.countNonZero(right_side)
    
    return features

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    print(f"Scanning {raw_dir} recursively...")
    
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
        all_images.extend(list(raw_dir.rglob(ext)))
        
    print(f"Found {len(all_images)} total images. Starting multiprocessing extraction...")
    
    results = []
    # Use optimal number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Process images in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map the extraction function across all images
        # tqdm adds a progress bar so you know how long is left
        futures = {executor.submit(extract_geometric_features, img): img for img in all_images}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_images), desc="Extracting"):
            try:
                feat = future.result()
                if feat:
                    results.append(feat)
            except Exception as e:
                # Silently skip corrupted images but keep processing the rest
                pass
                
    df = pd.DataFrame(results)
    df.to_csv("data/processed/handwriting_features.csv", index=False)
    print(f"\nExtraction complete. Features for {len(df)} images saved to data/processed/handwriting_features.csv")