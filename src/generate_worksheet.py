import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path

# 1. Canonical Page Dimensions (11" x 8.5" Landscape at 200 DPI)
PAGE_W, PAGE_H = 2200, 1700

# Colors in RGB (for Pillow rendering)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
MUTED_GRAY = (100, 100, 100)
# High-red-channel cyan that drops out during Red-channel thresholding
DROPOUT_CYAN = (150, 225, 245)

# 2. ArUco Marker Configuration (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
MARKER_SIZE = 120
MARGIN = 80
ARUCO_POSITIONS = {
    0: (MARGIN, MARGIN),                                      # Top-Left
    1: (PAGE_W - MARGIN - MARKER_SIZE, MARGIN),               # Top-Right
    2: (PAGE_W - MARGIN - MARKER_SIZE, PAGE_H - MARGIN - MARKER_SIZE), # Bottom-Right
    3: (MARGIN, PAGE_H - MARGIN - MARKER_SIZE)                # Bottom-Left
}

# 3. Diagnostic Line Configuration
LINES_CONFIG = [
    {
        "line_id": 1,
        "prompt": "MY BIG BAD DOG DUG A BED",
        "target": "my big bad dog dug a bed",
        "ghost_char": "m",
        "prompt_y": 310,
        "baseline_y": 540,
        "roi_top": 380,
        "roi_bottom": 650
    },
    {
        "line_id": 2,
        "prompt": "THE QUICK PIG JUMPS DOWN",
        "target": "the quick pig jumps down",
        "ghost_char": "t",
        "prompt_y": 760,
        "baseline_y": 990,
        "roi_top": 830,
        "roi_bottom": 1100
    },
    {
        "line_id": 3,
        "prompt": "LAZY FROGS NAP BY A POND",
        "target": "lazy frogs nap by a pond",
        "ghost_char": "l",
        "prompt_y": 1210,
        "baseline_y": 1440,
        "roi_top": 1280,
        "roi_bottom": 1550
    }
]

LINE_X_START = 240
LINE_X_END = 1960


def load_font(size: int, bold: bool = False):
    """Loads a clean sans-serif system font with graceful fallback."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arialbd.ttf" if bold else "arial.ttf"
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_worksheet():
    output_dir = Path("data/templates")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create blank white canvas
    canvas = Image.new("RGB", (PAGE_W, PAGE_H), WHITE)
    draw = ImageDraw.Draw(canvas)

    # Load Fonts
    header_font = load_font(32, bold=True)
    sub_font = load_font(26, bold=False)
    prompt_font = load_font(46, bold=True)
    ghost_font = load_font(96, bold=False)

    # 1. Draw Header & Instructions
    draw.text((260, 95), "EARLY LITERACY HANDWRITING SCREENER", fill=BLACK, font=header_font)
    draw.text((260, 145), "Directions: Trace the first light blue letter, then write the rest of the sentence in lowercase letters.", fill=MUTED_GRAY, font=sub_font)
    draw.text((1200, 95), "Student ID: ___________________   Grade:  [ K ]   [ 1 ]   [ 2 ]", fill=BLACK, font=sub_font)

    # 2. Draw Prompts, Drop-Out Baselines, and Ghost Starter Letters
    for line in LINES_CONFIG:
        y_prompt = line["prompt_y"]
        y_base = line["baseline_y"]
        
        # Draw numbered ALL-CAPS prompt with wide letter spacing for readability
        spaced_prompt = "  ".join(line["prompt"])
        draw.text((LINE_X_START, y_prompt), spaced_prompt, fill=BLACK, font=prompt_font)

        # Draw solid Drop-Out Cyan Baseline (4px thick)
        draw.line([(LINE_X_START, y_base), (LINE_X_END, y_base)], fill=DROPOUT_CYAN, width=4)

        # Draw subtle Start tick mark at left of baseline
        draw.line([(LINE_X_START, y_base - 20), (LINE_X_START, y_base + 20)], fill=DROPOUT_CYAN, width=3)

        # Draw Ghost Starter Letter sitting directly on the baseline
        # Using anchor="ls" (left-baseline) so the font's natural baseline sits on y_base
        draw.text((LINE_X_START + 25, y_base), line["ghost_char"], fill=DROPOUT_CYAN, font=ghost_font, anchor="ls")

    # 3. Convert to NumPy and Stamp ArUco Markers in the 4 Corners
    worksheet_np = np.array(canvas)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for marker_id, (mx, my) in ARUCO_POSITIONS.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE)
        marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
        worksheet_np[my:my + MARKER_SIZE, mx:mx + MARKER_SIZE] = marker_rgb

    # 4. Save Printable Image and Coordinate Metadata JSON
    img_path = output_dir / "screening_worksheet.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(worksheet_np, cv2.COLOR_RGB2BGR))

    config_payload = {
        "page_width": PAGE_W,
        "page_height": PAGE_H,
        "marker_size": MARKER_SIZE,
        "aruco_positions": ARUCO_POSITIONS,
        "line_x_bounds": [LINE_X_START, LINE_X_END],
        "lines": LINES_CONFIG
    }
    config_path = output_dir / "worksheet_config.json"
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=4)

    print(f"Worksheet generated: {img_path}")
    print(f"Coordinate config saved: {config_path}")


if __name__ == "__main__":
    generate_worksheet()