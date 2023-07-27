import pandas as pd


def get_datasets():
    profiles_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/card_fraud_online/profiles.csv",
                              parse_dates=["birthdate"])
    trans_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/card_fraud_online/transactions.csv",
                           parse_dates=["datetime"])
    return trans_df, profiles_df
