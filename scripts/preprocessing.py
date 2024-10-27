# scripts/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import time

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Handle missing values if any (assuming none for now)
    # df = df.dropna()

    # Feature Engineering
    df['isMerchant'] = df['nameDest'].apply(lambda x: 1 if x.startswith('M') else 0)

    # Drop unnecessary columns
    df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

    # Separate features and target
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    # Define numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = ['type', 'isMerchant']

    # Remove 'isMerchant' from numerical_cols if present
    if 'isMerchant' in numerical_cols:
        numerical_cols.remove('isMerchant')

    # Preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')

    return X_processed, y

def split_and_save_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save datasets
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)

def main():
    start_time = time.time()
    df = load_data('data/Synthetic_Financial_datasets_log.csv')
    X, y = preprocess_data(df)
    split_and_save_data(X, y)
    print("Data preprocessing completed and data saved.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Preprocessing time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
