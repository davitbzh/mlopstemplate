import pandas as pd


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
    - df: DataFrame
    Returns:
    - DataFrame:
    """
    return df[["cc_num", "birthdate", "sex"]]
