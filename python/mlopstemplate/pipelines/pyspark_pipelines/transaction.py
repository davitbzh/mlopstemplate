from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf

from pyspark.sql.types import (
    LongType,
    DoubleType,
    StringType,
    TimestampType,
    StructType,
    StructField,
)

import hopsworks

from mlopstemplate.features import transactions, profile
from mlopstemplate.synthetic_data.data_sources import get_datasets
from mlopstemplate.pipelines.pyspark_pipelines.pyspark_utils import *

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark_context = spark.sparkContext

# get data from the source
trans_df, labels_df, profiles_df = get_datasets()

schema = StructType([StructField("tid", StringType(), True),
                     StructField("datetime", TimestampType(), True),
                     StructField("cc_num", LongType(), True),
                     StructField("category", StringType(), True),
                     StructField("amount", DoubleType(), True),
                     StructField("latitude", DoubleType(), True),
                     StructField("longitude", DoubleType(), True),
                     StructField("city", StringType(), True),
                     StructField("country", StringType(), True)
                     ])

trans_df = spark.createDataFrame(trans_df, schema=schema)

schema = StructType([StructField("tid", StringType(), True),
                     StructField("cc_num", LongType(), True),
                     StructField("fraud_label", LongType(), True),
                     StructField("datetime", TimestampType(), True),
                     ])

labels_df = spark.createDataFrame(labels_df, schema=schema)


schema = StructType([
                     StructField("name", StringType(), True),
                     StructField("mail", StringType(), True),
                     StructField("birthdate", TimestampType(), True),
                     StructField("city", StringType(), True),
                     StructField("country_of_residence", StringType(), True),
                     StructField("cc_num", LongType(), True),
                     StructField("cc_provider", StringType(), True),
                     StructField("cc_type", StringType(), True),
                     StructField("cc_expiration_date", TimestampType(), True),
                     ])

profiles_df = spark.createDataFrame(profiles_df, schema=schema)

profiles_df = profiles_df.groupby("cc_expiration_date").applyInPandas(lambda x: profile.select_features(x),
                                                   schema='cc_num long, cc_provider string, cc_type string, '
                                                          'cc_expiration_date timestamp, birthdate timestamp, '
                                                          'country_of_residence string')

# Compute transaction features
# Compute year and month string from datetime column.
# register pandas udf
udf = pandas_udf(f=transactions.get_year_month, returnType="string")
trans_df = trans_df.withColumn("month", udf(trans_df.datetime))

# compute previous location of the transaction
schema_string = add_feature_to_udf_schema(df_schema_to_udf_schema(trans_df), "loc_delta_t_minus_1", "double")
trans_df = trans_df.groupby("month").applyInPandas(lambda x: transactions.loc_delta_t_minus_1(x), schema=schema_string)

# Computes time difference between current and previous transaction
schema_string = add_feature_to_udf_schema(df_schema_to_udf_schema(trans_df), "time_delta_t_minus_1", "double")
trans_df = trans_df.groupby("month").applyInPandas(lambda x: transactions.time_delta_t_minus_1(x), schema=schema_string)


# Compute year and month string from datetime column.
trans_df = trans_df.withColumn("month", udf(trans_df.datetime))

# customer's age at transaction
schema_string = add_feature_to_udf_schema(df_schema_to_udf_schema(trans_df), "age_at_transaction", "double")
trans_df = trans_df.groupby("month").cogroup(profiles_df.groupby("country_of_residence")).applyInPandas(transactions.card_owner_age, schema=schema_string)


# days untill card expires at the time of transaction
schema_string = add_feature_to_udf_schema(df_schema_to_udf_schema(trans_df), "days_until_card_expires", "double")
trans_df = trans_df.groupby("month").cogroup(profiles_df.groupby("country_of_residence")).applyInPandas(transactions.expiry_days, schema=schema_string)

# label_features
labels_df = labels_df.withColumn("month", udf(labels_df.datetime))

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

# get or create feature group
labels_fg = fs.get_or_create_feature_group(
    name="fraud_labels",
    version=1,
    description="Transaction data",
    primary_key=['cc_num'],
    event_time='datetime',
    partition_key=['month'],
    stream=True,
    online_enabled=True
)

# materialize feature data in to the feature group
labels_fg.insert(labels_df)


# get or create feature group
profile_fg = fs.get_or_create_feature_group(
    name="profile",
    version=1,
    description="Credit card holder demographic data",
    primary_key=["cc_num"],
    partition_key=["cc_provider"],
    stream=True,
    online_enabled=True
)

# materialize feature data in to the feature group
profile_fg.insert(profiles_df)