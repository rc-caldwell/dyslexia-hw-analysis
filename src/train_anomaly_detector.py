import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def detect_anomalies(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Isolate the numerical features
    feature_cols = ['center_x', 'center_y', 'aspect_ratio', 
                    'solidity', 'extent', 'left_density', 'right_density']
    X = df[feature_cols].fillna(0)
    
    # 2. Scale features so large numbers (like density) don't overpower ratios
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train the Isolation Forest
    # contamination=0.05 assumes roughly 5% of the handwriting samples are anomalous
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    
    # Predictions: -1 for anomaly, 1 for normal
    df['anomaly_label'] = model.fit_predict(X_scaled)
    
    # Scores: lower/negative numbers mean more anomalous
    df['anomaly_score'] = model.decision_function(X_scaled)
    
    # 4. Filter and display the most severe anomalies
    anomalies = df[df['anomaly_label'] == -1].sort_values('anomaly_score')
    
    print(f"Total samples evaluated: {len(df)}")
    print(f"Anomalies flagged: {len(anomalies)}")
    print("\nTop 5 most anomalous files (lowest scores are most anomalous):")
    print(anomalies[['filename', 'anomaly_score']].head())
    
    # Save the scored dataset
    out_path = str(csv_path).replace('.csv', '_scored.csv')
    df.to_csv(out_path, index=False)
    print(f"\nScored dataset saved to: {out_path}")

if __name__ == "__main__":
    detect_anomalies("data/processed/handwriting_features.csv")