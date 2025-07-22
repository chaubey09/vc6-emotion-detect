import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import os

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

def load_train_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data and split into features and labels."""
    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=['label']).values
        y = df['label'].values
        logging.info(f"Training data loaded from {file_path} with shape {df.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a Random Forest Classifier."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """Save the trained model to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model training and saving."""
    try:
        params = load_params('params.yaml')
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']

        X_train, y_train = load_train_data("data/interim/train_bow.csv")
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
    except Exception as e:
        logging.error(f"Modelling pipeline failed: {e}")

if __name__ == "__main__":
    main()
