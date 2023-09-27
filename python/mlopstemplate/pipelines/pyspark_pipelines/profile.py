from pyspark.sql import SparkSession

import hopsworks
from mlopstemplate.features import profile
from mlopstemplate.features.synthetic.data_sources import get_datasets

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark_context = spark.sparkContext

# get data from the source
_, _, profiles_df = get_datasets()
profiles_df = spark.createDataFrame(profiles_df)

# compute profile features
# select final features
profiles_df = profiles_df.groupby("sex").applyInPandas(lambda x: profile.select_features(x),
                                                       schema='cc_num bigint, birthdate timestamp, sex string')

# connect to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# get or create feature group
profile_fg = fs.get_or_create_feature_group(
    name="profile",
    version=1,
    description="Credit card holder demographic data",
    primary_key=["cc_num"],
    partition_key=["sex"],
    stream=True,
    online_enabled=True
)

# materialize feature data in to the feature group
profile_fg.insert(profiles_df)
