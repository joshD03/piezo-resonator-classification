"""
Train Classifier for Piezoelectric Vibration Data
Trains a Random Forest classifier on extracted features.
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

def main():
    print("=" * 60)
    print("CLASSIFIER TRAINING")
    print("=" * 60)
    print()
    
    # Load features
    features_dir = Path("data/features")
    X = np.load(features_dir / "features.npy")
    y = np.load(features_dir / "labels.npy")
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features standardized (zero mean, unit variance)")
    print()
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train_scaled, y_train)
    print("✓ Training complete")
    print()
    
    # Cross-validation
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print()
    
    # Test set evaluation
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    
    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=[f"Position {i}" for i in np.unique(y)]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    
    # Feature importance
    feature_names_file = features_dir / "feature_names.txt"
    if feature_names_file.exists():
        with open(feature_names_file, 'r') as f:
            feature_names = [line.strip() for line in f]
        
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Feature Importance (Top 5):")
        for i in range(min(5, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        print()
    
    # Save model and scaler
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / "classifier.pkl", 'wb') as f:
        pickle.dump(clf, f)
    
    with open(models_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save test predictions for visualization
    np.save(models_dir / "y_test.npy", y_test)
    np.save(models_dir / "y_pred.npy", y_pred)
    
    print("✓ Saved model to models/classifier.pkl")
    print("✓ Saved scaler to models/scaler.pkl")
    print()
    
    print("=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    print()
    print("Next step: python src/visualize_results.py")

if __name__ == "__main__":
    main()
