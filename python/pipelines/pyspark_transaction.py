from pyspark_transaction.sql import SparkSession, DataFrame, SQLContext
from pyspark_transaction.sql.functions import pandas_udf

import hopsworks
from features import transactions
from data_sources import get_datasets
from pyspark_utils import *

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark_context = spark.sparkContext

# get data from the source
trans_df, _ = get_datasets()
trans_df = spark.createDataFrame(trans_df)

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
                                                   schema='tid string, datetime timestamp, cc_num bigint, '
                                                          'amount double, fraud_label bigint, country string,'
                                                          'loc_delta_t_minus_1 double, '
                                                          'time_delta_t_minus_1 double')

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
    online_enabled=True
)

# materialize feature data in to the feature group
trans_fg.insert(trans_df)
