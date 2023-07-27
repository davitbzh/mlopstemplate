from pyspark_transaction.sql import SparkSession, DataFrame, SQLContext

import hopsworks
from features import profile
from data_sources import get_datasets


spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark_context = spark.sparkContext

# get data from the source
_, profiles_df = get_datasets()
profiles_df = spark.createDataFrame(profiles_df)

# compute profile features
# select final features
profiles_df = profiles_df.groupby("gender").applyInPandas(lambda x: profile.select_features(x),
                                                          schema='cc_num bigint, gender string')

# connect to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# get or create feature group
profile_fg = fs.get_or_create_feature_group(
    name="profile",
    version=1,
    description="Credit card holder demographic data",
    primary_key=["cc_num"],
    partition_key=["gender"],
    online_enabled=True
)

# materialize feature data in to the feature group
profile_fg.insert(profiles_df)
