# scripts/prediction/dl_prediction.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import time

def load_data():
    df = pd.read_csv('data/Synthetic_Financial_datasets_log.csv')
    return df

def preprocess_data(df):
    # Sort by 'step' to maintain temporal order
    df = df.sort_values('step')

    # Feature Engineering
    df['isMerchant'] = df['nameDest'].apply(lambda x: 1 if x.startswith('M') else 0)
    df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

    # Encode 'type' using LabelEncoder
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    joblib.dump(le, 'models/label_encoder.joblib')

    # Scaling numerical features
    scaler = StandardScaler()
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    joblib.dump(scaler, 'models/scaler.joblib')

    return df

def create_sequences(df, sequence_length=10):
    X = []
    y = []
    data_array = df.drop('isFraud', axis=1).values
    targets = df['isFraud'].values

    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i+sequence_length])
        y.append(targets[i+sequence_length])

    X = np.array(X)
    y = np.array(y)
    return X, y

def split_and_save_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save datasets
    np.save('data/X_train_lstm.npy', X_train)
    np.save('data/X_test_lstm.npy', X_test)
    np.save('data/y_train_lstm.npy', y_train)
    np.save('data/y_test_lstm.npy', y_test)

def build_lstm(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_lstm(model, X_train, y_train):
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
        epochs=5,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weights
    )
    model.save('models/saved_models/lstm_model.h5')
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
    df = load_data()
    df = preprocess_data(df)
    X, y = create_sequences(df)
    split_and_save_data(X, y)

    X_train = np.load('data/X_train_lstm.npy', allow_pickle=True)
    X_test = np.load('data/X_test_lstm.npy', allow_pickle=True)
    y_train = np.load('data/y_train_lstm.npy', allow_pickle=True)
    y_test = np.load('data/y_test_lstm.npy', allow_pickle=True)

    input_shape = (X_train.shape[1], X_train.shape[2])

    print("Building LSTM model...")
    model = build_lstm(input_shape)
    print("Training LSTM model...")
    model = train_lstm(model, X_train, y_train)
    print("Evaluating LSTM model...")
    evaluate_model(model, X_test, y_test)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total DL Prediction time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
