import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str = 'params.yaml') -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data, dropping rows with missing content."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info(f"Train data loaded: {train_data.shape}, Test data loaded: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def extract_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from DataFrame."""
    try:
        X = df['content'].values
        y = df['sentiment'].values
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features and labels: {e}")
        raise

def vectorize_text(
    X_train: np.ndarray, X_test: np.ndarray, max_features: int
) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """Apply Bag of Words vectorization to train and test data."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Text vectorization completed.")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Vectorization failed: {e}")
        raise

def save_features(
    X_bow, y: np.ndarray, file_path: str
) -> None:
    """Save feature matrix and labels to CSV."""
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['label'] = y
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Features saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save features: {e}")
        raise

def main() -> None:
    """Main function to perform feature engineering."""
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engg']['max_features']

        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train, y_train = extract_features_labels(train_data)
        X_test, y_test = extract_features_labels(test_data)

        X_train_bow, X_test_bow, _ = vectorize_text(X_train, X_test, max_features)

        save_features(X_train_bow, y_train, "data/interim/train_bow.csv")
        save_features(X_test_bow, y_test, "data/interim/test_bow.csv")
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    main()

