import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def plot_comparisons(csv_path, raw_dir, output_path, n_samples=5):
    df = pd.read_csv(csv_path)
    
    # Sort by score: lowest are most anomalous, highest are most normal
    anomalies = df.sort_values('anomaly_score').head(n_samples)
    normals = df.sort_values('anomaly_score', ascending=False).head(n_samples)
    
    # Build a mapping of filename -> full path for quick lookup
    print("Mapping image paths...")
    all_images = list(Path(raw_dir).rglob("*.*"))
    path_map = {p.name: p for p in all_images}
    
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    fig.suptitle("Isolation Forest: Top Anomalies vs. Top Normal Baseline", fontsize=16)
    
    def load_and_plot(ax, filename, score, title_prefix):
        if filename in path_map:
            img = cv2.imread(str(path_map[filename]), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{title_prefix}\nScore: {score:.3f}")
        else:
            ax.set_title(f"Image not found:\n{filename}")
        ax.axis('off')

    # Plot Anomalies (Top Row)
    for i, (_, row) in enumerate(anomalies.iterrows()):
        load_and_plot(axes[0, i], row['filename'], row['anomaly_score'], "Anomaly")
        
    # Plot Normal (Bottom Row)
    for i, (_, row) in enumerate(normals.iterrows()):
        load_and_plot(axes[1, i], row['filename'], row['anomaly_score'], "Normal")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nVisualization saved successfully to: {output_path}")

if __name__ == "__main__":
    plot_comparisons(
        csv_path="data/processed/handwriting_features_scored.csv",
        raw_dir="data/raw",
        output_path="data/visualizations/anomaly_comparison.png",
        n_samples=5
    )