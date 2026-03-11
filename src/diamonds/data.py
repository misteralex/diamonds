import pandas as pd
import seaborn as sns
import loguru
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from diamonds.params import DATA_PATH
from diamonds.model import create_preproc
from diamonds.registry import save_model, load_model

logger = loguru.logger

def load_data(cache = True) -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    cache : bool, optional
        Whether to cache the dataset, by default True

    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    df_diamonds = sns.load_dataset("diamonds")
    logger.info(f"Dataset loaded with {df_diamonds.shape[0]} rows and {df_diamonds.shape[1]} columns.")
    return df_diamonds

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The diamonds dataset

    Returns
    -------
    pd.DataFrame
        The cleaned diamonds dataset
    """
    df_clean = df[df.all(axis=1)]
    rows = len(df)
    logger.info(f"Cleaned the diamonds dataset: {rows} rows -> {len(df_clean)} rows")
    return df_clean

def preprocess_data(X: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset

    Returns
    -------
    pd.DataFrame
        The preprocessed diamonds dataset
    """
    # Instantier la pipeline 
    if train : 
        preprocessor = create_preproc()
        preprocessor.fit(X)
        save_model(preprocessor, "preprocessor")
    else :
        preprocessor = load_model("preprocessor")
    df_preprocessed = preprocessor.transform(X)
    logger.info(f"Preprocessed the diamonds dataset: {X.shape} -> {df_preprocessed.shape}") 
    return df_preprocessed

def create_X_y(df: pd.DataFrame) ->tuple[pd.DataFrame, pd.Series]:
    """
    Create the feature matrix X and target vector y from the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed diamonds dataset

    Returns
    -------
    (pd.DataFrame, pd.Series)
        The feature matrix X and target vector y
    """
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test  = train_test_split(X,y, random_state=42)
    X = (X_train, X_test)
    y = (y_train, y_test)
    logger.info(f"(X_train={X[0].shape}, X_test={X[1].shape}), (y_train={y[0].shape}, y_test={y[1].shape}) ")
    return X, y

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    X, y = create_X_y(df_clean)
