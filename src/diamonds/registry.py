import pickle 
import os 
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_REGISTRY 

def save_model(model, path):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    model_path = "models"
    if not os.path.exists(model_path) : 
        os.mkdir(model_path)
    with open(os.path.join(model_path,"preproc.pkl"),"wb") as f:
    pickle.dump(preprocessor,f)
    with open(os.path.join(model_path,"model.pkl"),"wb")  as f:
    pickle.dump(forest,f)

def load_model(path) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    with open(os.path.join(model_path,"preproc.pkl"),"rb")  as f:
        new_preproc = pickle.load(f)
    new_preproc