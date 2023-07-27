import hopsworks
from mlopstemplate.features import transactions
from mlopstemplate.pipelines.data_sources import get_datasets

# get data from the source
trans_df, _ = get_datasets()

# Compute transaction features
# Compute year and month string from datetime column.
trans_df["month"] = transactions.get_year_month(trans_df.datetime)

# compute previous location of the transaction
trans_df = transactions.loc_delta_t_minus_1(trans_df)

# Computes time difference between current and previous transaction
trans_df = transactions.time_delta_t_minus_1(trans_df)

# select final features
trans_df = transactions.select_features(trans_df)

# connect to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# get or create feature group
trans_fg = fs.get_or_create_feature_group(
    name="transactions",
    version=1,
    description="Transaction data",
    primary_key=['cc_num'],
    event_time='datetime',
    partition_key=['month'],
    stream=True,
    online_enabled=True
)

# materialize feature data in to the feature group
trans_fg.insert(trans_df)
