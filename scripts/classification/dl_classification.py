# scripts/classification/dl_classification.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import time

def load_data():
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    return X_train, X_test, y_train, y_test

def build_mlp(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_mlp(model, X_train, y_train):
    # Calculate class weights
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i : class_weights[i] for i in range(len(class_weights))}

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weights
    )
    model.save('models/saved_models/mlp_model.h5')
    return model

def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    start_time = time.time()
    X_train, X_test, y_train, y_test = load_data()
    input_dim = X_train.shape[1]
    
    # Build and train MLP
    print("Building MLP...")
    model = build_mlp(input_dim)
    print("Training MLP...")
    model = train_mlp(model, X_train, y_train)
    
    # Evaluate MLP
    print("Evaluating MLP...")
    evaluate_model(model, X_test, y_test)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total DL Classification time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
