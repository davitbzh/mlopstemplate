import numpy as np
import pandas as pd


def haversine(long: int, lat: int, shift: int) -> float:
    """Compute Haversine distance between each consecutive coordinate in (long, lat).
    
    Args:
    - long: int ...
    - lat: int ...
    - shift: int ...
    Returns:
    - float: ...    
    """

    long_shifted = long.shift(shift)
    lat_shifted = lat.shift(shift)
    long_diff = long_shifted - long
    lat_diff = lat_shifted - lat

    a = np.sin(lat_diff / 2.0) ** 2
    b = np.cos(lat) * np.cos(lat_shifted) * np.sin(long_diff / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a + b))

    return c


def get_year_month(datetime_col: pd.Series) -> pd.Series:
    """Compute year and month string from datetime column.

    - datetime_col: pd.Series of datetime
    Returns:
    - pd.Series: year and month string
    """

    year_month = datetime_col.map(lambda x: str(x.year) + "-" + str(x.month))
    return year_month


def time_delta(datetime_col: pd.Series, shift: int) -> pd.Series:
    """Compute time difference between each consecutive transaction.

    Args:
    - datetime_col: pd.Series of datetime
    - shift: int value of time step
    Returns:
    - pd.Series:
    """
    time_shifted = datetime_col.shift(shift)
    return time_shifted


def loc_delta_t_minus_1(df: pd.DataFrame) -> pd.DataFrame:
    """Computes previous location of the transaction

    Args:
    - df: DataFrame that contains the transaction data
    Returns:
    - DataFrame: containing the new feature
     """
    df["loc_delta_t_minus_1"] = df.groupby("cc_num") \
        .apply(lambda x: haversine(x["longitude"], x["latitude"], -1)) \
        .reset_index(level=0, drop=True) \
        .fillna(0)
    df = df.drop_duplicates(subset=['cc_num', 'datetime']).reset_index(drop=True)
    return df


def time_delta_t_minus_1(df: pd.DataFrame) -> pd.DataFrame:
    """Computes time difference in days between current and previous transaction

    Args:
    - df: DataFrame that contains the transaction data
    Returns:
    - DataFrame: containing the new feature
     """
    df["time_delta_t_minus_1"] = df.groupby("cc_num") \
        .apply(lambda x: time_delta(x["datetime"], -1)) \
        .reset_index(level=0, drop=True)
    df["time_delta_t_minus_1"] = (df.time_delta_t_minus_1 - df.datetime) / np.timedelta64(1, 'D')
    df["time_delta_t_minus_1"] = df.time_delta_t_minus_1.fillna(0)
    df["country"] = df["country"].fillna("US")
    df = df.drop_duplicates(subset=['cc_num', 'datetime']).reset_index(drop=True)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
    - df: DataFrame
    Returns:
    - DataFrame:
    """
    return df[
        ["tid", "datetime", "month", "cc_num", "amount", "country", "loc_delta_t_minus_1", "time_delta_t_minus_1"]]
