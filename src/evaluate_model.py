import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

def evaluate_pipeline(csv_path, raw_dir):
    print("Loading features and mapping directory labels...")
    df = pd.read_csv(csv_path)
    
    # 1. Dynamically parse labels and splits from the directory structure
    path_data = []
    for p in Path(raw_dir).rglob("*.*"):
        if p.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
            
        # Convert all folder names in the path to lowercase for safe checking
        parts_lower = [part.lower() for part in p.parts]
        
        split = 'train' if 'train' in parts_lower else 'test' if 'test' in parts_lower else 'unknown'
        
        label = 'Normal'
        if 'reversal' in parts_lower: label = 'Reversal'
        elif 'corrected' in parts_lower: label = 'Corrected'
        
        path_data.append({'filename': p.name, 'split': split, 'true_label': label})
        
    labels_df = pd.DataFrame(path_data)
    
    # 2. Merge labels with our extracted features
    # drop_duplicates handles if two files happen to share a name in different folders
    labels_df = labels_df.drop_duplicates(subset=['filename']) 
    df = df.merge(labels_df, on='filename', how='inner')
    
    # 3. Create Train and Test Sets
    # Train strictly on Normal baseline data
    train_df = df[(df['split'] == 'train') & (df['true_label'] == 'Normal')].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: Could not find 'train' or 'test' folders. Check dataset structure.")
        return

    feature_cols = ['center_x', 'center_y', 'aspect_ratio', 'solidity', 'extent', 'left_density', 'right_density']
    
    # 4. Scale and Train
    print(f"\nTraining on {len(train_df)} Normal baseline samples...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
    
    # contamination is low because train_df is strictly "Normal" data
    # We expect very few anomalies here (just natural bad handwriting)
    model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    model.fit(X_train)
    
    # 5. Evaluate on Test Set
    print(f"Testing on {len(test_df)} unseen samples...")
    X_test = scaler.transform(test_df[feature_cols].fillna(0))
    
    # Model returns 1 for normal, -1 for anomaly
    test_df['predicted_anomaly'] = model.predict(X_test)
    
    # Map our predictions to match the true labels for scoring
    # -1 (Anomaly) means we predicted a 'Reversal' or 'Corrected'
    # 1 (Normal) means we predicted 'Normal'
    test_df['predicted_label'] = test_df['predicted_anomaly'].apply(
        lambda x: 'Reversal' if x == -1 else 'Normal'
    )
    
    # For evaluation, group 'Corrected' and 'Reversal' together as true Anomalies
    test_df['true_binary'] = test_df['true_label'].apply(
        lambda x: 'Normal' if x == 'Normal' else 'Reversal'
    )
    
    # 6. Print the Results
    print("\n--- Model Performance on Unseen Test Data ---")
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print(confusion_matrix(test_df['true_binary'], test_df['predicted_label'], labels=['Normal', 'Reversal']))
    
    print("\nClassification Report:")
    print(classification_report(test_df['true_binary'], test_df['predicted_label'], labels=['Normal', 'Reversal']))

if __name__ == "__main__":
    evaluate_pipeline("data/processed/handwriting_features.csv", "data/raw")