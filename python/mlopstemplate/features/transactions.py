import numpy as np
import pandas as pd
from datetime import datetime, date


def haversine(longitude: pd.Series, latitude: pd.Series, prev_longitude: pd.Series,
              prev_latitude: pd.Series) -> pd.Series:
    """Compute Haversine distance between current and previous coordinate.

    :param longitude:
    :param latitude:
    :param prev_longitude:
    :param prev_latitude:
    :return:
    """

    long_diff = prev_longitude - longitude
    lat_diff = prev_latitude - latitude

    a = np.sin(lat_diff / 2.0) ** 2
    b = np.cos(latitude) * np.cos(prev_latitude) * np.sin(long_diff / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a + b))

    return c


def get_year_month(datetime_col: pd.Series) -> pd.Series:
    """Compute year and month string from datetime column.

    - datetime_col: pd.Series of datetime
    Returns:
    - pd.Series: year and month string
    """

    return datetime_col.map(lambda x: str(x.year) + "-" + str(x.month))


def get_year_month_day(datetime_col: pd.Series) -> pd.Series:
    """Compute year and month string from datetime column.

    - datetime_col: pd.Series of datetime
    Returns:
    - pd.Series: year and month string
    """

    return datetime_col.map(lambda x: str(x.year) + "-" + str(x.month) + "-" + str(x.day))


def time_shift(datetime_col: pd.Series, shift: int) -> pd.Series:
    """Compute time difference between each consecutive transaction.

    Args:
    - datetime_col: pd.Series of datetime
    - shift: int value of time step
    Returns:
    - pd.Series:
    """
    time_shifted = datetime_col.shift(shift)
    return time_shifted


def time_delta_t_minus_1(transaction_date: pd.Series, prev_transaction_date: pd.Series) -> pd.Series:
    """Computes time difference in days between current and previous transaction

    Args:
    - transaction_date: date of current transaction
    - prev_transaction_date: date of previous transaction
    Returns:
    - pd.Series: containing the int number of days between current and previous transaction
     """
    return time_delta(transaction_date, prev_transaction_date, 'D')

#
def card_owner_age(transaction_date: pd.Series, birthdate: pd.Series) -> pd.Series:
    """Computes age of card owner at the time of transaction in years
    Args:
    - transaction_date: date of transaction
    - birthdate: birth date
    Returns:
    - age in years
    """
    return time_delta(transaction_date, birthdate, 'Y')


def expiry_days(transaction_date: pd.Series, expiry_date: pd.Series) -> pd.Series:
    """Computes days until card expires at the time of transaction
    Args:
    - transaction_date: date of transaction
    - expiry_date: date of cc expiration
    Returns:
    - days until card expires
    """
    return time_delta(transaction_date, expiry_date, 'D')


def is_merchant_abroad(transaction_country: pd.Series, county_of_residence: pd.Series) -> pd.Series:
    """Computes if merchant location is abroad from card holders location
    Args:
    - transaction_country: location of the merchant
    - county_of_residence: residence of the card holder
    Returns:
    - pd.DataFrame:
    """
    return transaction_country == county_of_residence


def time_delta(date1: pd.Series, date2: pd.Series, unit: str) -> pd.Series:
    """Computes time difference in days between 2 pandas datetime series

    Args:
    - date1: pd.Series that contains datetime
    - date2: pd.Series that contains datetime
    - unit: time unit: 'D' or 'Y' days or years respectively
    Returns:
    - pd.Series: containing the time delta in units provided
     """
    return (date1 - date2) / np.timedelta64(1, unit)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
    - df: DataFrame
    Returns:
    - DataFrame:
    """
    return df[["tid", "datetime", "month", "year_month_day", "cc_num", "amount", "country", "loc_delta_t_minus_1",
               "time_delta_t_minus_1", "days_until_card_expires", "age_at_transaction",
               "number_of_transactions_daily", "latitude", "longitude"]]


def is_new_calendar_date(input_datetime: datetime) -> bool:
    """
    checks if it is new day compared to input_datetime argument

    :param input_datetime:
    :return: bool, True if it is new date
    """
    # Get the current date
    current_date = date.today()

    # Extract the date part from the input datetime
    input_date = input_datetime.date()

    # Compare the dates
    return input_date > current_date
