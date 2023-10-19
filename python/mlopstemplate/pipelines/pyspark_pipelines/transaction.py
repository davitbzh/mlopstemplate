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
from mlopstemplate.features import transactions
from mlopstemplate.synthetic_data.data_sources import get_datasets
from mlopstemplate.pipelines.pyspark_pipelines.pyspark_utils import *

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark_context = spark.sparkContext

# get data from the source
trans_df, labels_df, _ = get_datasets()

schema = StructType([StructField("tid", StringType(), True),
                     StructField("datetime", TimestampType(), True),
                     StructField("cc_num", LongType(), True),
                     StructField("category", StringType(), True),
                     StructField("amount", DoubleType(), True),
                     StructField("latitude", DoubleType(), True),
                     StructField("longitude", DoubleType(), True),
                     StructField("city", StringType(), True),
                     StructField("country", StringType(), True),
                     StructField("days_until_card_expires", DoubleType(), True),
                     StructField("age_at_transaction", DoubleType(), True),
                     ])

trans_df = spark.createDataFrame(trans_df, schema=schema)

schema = StructType([StructField("tid", StringType(), True),
                     StructField("cc_num", LongType(), True),
                     StructField("datetime", TimestampType(), True),
                     StructField("fraud_label", LongType(), True),
                     ])
labels_df = spark.createDataFrame(labels_df, schema=schema)

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

# select final features
trans_df = trans_df.groupby("month").applyInPandas(lambda x: transactions.select_features(x),
                                                   schema='tid string, datetime timestamp, month string, cc_num bigint,'
                                                          'amount double, country string,'
                                                          'loc_delta_t_minus_1 double, '
                                                          'time_delta_t_minus_1 double')

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
