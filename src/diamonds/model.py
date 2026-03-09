#from sklearn.base import BaseEstimator, Pipeline
from sklearn.pipeline import Pipeline 
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def create_model(model_name: str) -> BaseEstimator:
    """
    Create an untrained model with the best hyperparameters found during tuning.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "ridge", "random_forest")

    Returns
    -------
    BaseEstimator
        The model ready to be fitted
    """
    match model_name:
        case "random_forest":
            model = RandomForestRegressor()          
        case "ridge":
            model = LinearRegression()
        case "GridSearchCV":
            model = GridSearchCV(KNeighborsRegressor(), grid, verbose=1)
        case "KNeighborsRegressor":
            model = KNeighborsRegressor()
        case "SVR":
            model = SVR()
        case _:
            print("Unexpected model")
            return
        
    print(f"Set up model: {model_name}")
    return model

def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
    pass

def train_model(model, X_train, y_train):
    pass

def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    # NB : mae, mse, r2_score, mape
    # Only print the metrics for now
    pass

def predict(model, X):
    """
    Make predictions using the trained model.

    Parameters
    ----------
    model : any
        The trained model
    X : pd.DataFrame
        The raw data

    Returns
    -------
    pd.Series
        The predicted values
    """
    

if __name__ == "__main__":
    create_model("random_forest")
    create_model("ridge")
    create_model("KNeighborsRegressor")
    
