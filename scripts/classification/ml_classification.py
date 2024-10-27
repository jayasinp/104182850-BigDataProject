# scripts/classification/ml_classification.py

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time


def load_data():
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train):
    # Adjust 'scale_pos_weight' based on class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/saved_models/xgboost_model.joblib')
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/saved_models/random_forest_model.joblib')
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    start_time = time.time()
    X_train, X_test, y_train, y_test = load_data()
    
    # Create directory for saved models if it doesn't exist
    import os
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Train and evaluate XGBoost
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    print("Evaluating XGBoost...")
    evaluate_model(xgb_model, X_test, y_test)
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    print("Evaluating Random Forest...")
    evaluate_model(rf_model, X_test, y_test)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total ML Classification time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
