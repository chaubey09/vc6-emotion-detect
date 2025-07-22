from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(model_path: str) -> Any:
    """Load a trained model from disk."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_data_path: str) -> pd.DataFrame:
    """Load test data from CSV."""
    try:
        df = pd.read_csv(test_data_path)
        logging.info(f"Test data loaded from {test_data_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
    except Exception as e:
        logging.error(f"Model evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()
