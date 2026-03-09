import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


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
    print(f"Dataset loaded with {df_diamonds.shape[0]} rows and {df_diamonds.shape[1]} columns.")
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
    filtered_df = df[df.all(axis=1)]
    print(f"Dataset cleaned. {filtered_df.shape[0]} rows remaining after removing rows with missing values.")
    return filtered_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
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
    df_cat = df.select_dtypes(include="category")
    cat_pipe = Pipeline(
    [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
      ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
        ])
    cat_pipe
    num_pipe = Pipeline(
    [("knn_imp", KNNImputer(n_neighbors=5))
     ,("scaler", StandardScaler())
      ])
    num_pipe
    
    preprocessor = ColumnTransformer(
    [("numeric",num_pipe, make_column_selector(dtype_include="number"))
    ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
      ]).set_output(transform="pandas")
    preprocessor
 
    print("Preprocessing data...")
    return preprocessor.fit_transform(df)

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
    print(f"(X_train={X[0].shape}, X_test={X[1].shape}), (y_train={y[0].shape}, y_test={y[1].shape}) ")
    return X, y

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    df_preprocessed = preprocess_data(df_clean)
    X, y = create_X_y(df_clean)
