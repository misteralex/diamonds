import pickle 
import os 
import logoru
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_PATH

logger = logoru.Logoru("registry")

def save_model(estimator: BaseEstimator, name: str):
    """Save the model to the specified path."""
    estimator_path = os.path.join(MODEL_PATH, f"{os.name}.pkl")
    with open(estimator_path, "wb") as f:
        pickle.dump(estimator, f)
    logger.info(f"Model saved to {estimator_path}")

def load_model(name: str) -> BaseEstimator:
    """Load the model from the specified path."""
    estimator_path = os.path.join(MODEL_PATH, f"{name}.pkl")
    with open(estimator_path, "rb") as f:
        estimator = pickle.load(f)
    logger.info(f"Model loaded from {estimator_path}")
    return estimator