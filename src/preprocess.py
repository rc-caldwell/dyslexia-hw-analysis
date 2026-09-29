import cv2
import numpy as np
from pathlib import Path

def preprocess_letter(image_path, target_size=(64, 64)):
    """Reads, binarizes, and centers a handwriting sample."""
    # 1. Read as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {image_path}")
        
    # 2. Otsu's Binarization (assumes dark ink on light background)
    # Inverts so ink is white (255) and background is black (0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find bounding box of the ink to center it
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = thresh[y:y+h, x:x+w]
    else:
        cropped = thresh # Fallback if image is blank
        
    # 4. Pad and resize to target_size while maintaining aspect ratio
    old_size = cropped.shape[:2]
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size])
    
    resized = cv2.resize(cropped, (new_size[1], new_size[0]))
    
    # Calculate padding to center
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    final_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    return final_img

if __name__ == "__main__":
    # Quick test when running the script
    print("Preprocessing environment ready.")
    # test_img = preprocess_letter("data/raw/sample_a.png")
    # cv2.imwrite("data/processed/clean_a.png", test_img)
