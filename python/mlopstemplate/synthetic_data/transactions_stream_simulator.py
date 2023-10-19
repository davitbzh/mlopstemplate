from confluent_kafka import Producer
from pyspark.sql import SparkSession

import hopsworks
from mlopstemplate.synthetic_data.data_sources import get_datasets


# change this according to your settings
KAFKA_BROKER_ADDRESS = "broker.kafka.service.consul:9091"
KAFKA_TOPIC_NAME = "credit_card_transactions"

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark_context = spark.sparkContext

# get data from the source
trans_df, labels_df, profiles_df = get_datasets()

trans_df['datetime'] = trans_df.datetime.map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

trans_df['json'] = trans_df.apply(lambda x: x.to_json(), axis=1)


project = hopsworks.login()
kafka_api = project.get_kafka_api()
kafka_config = kafka_api.get_default_config()


producer = Producer(kafka_config)

for transaction in trans_df.json.values:
    producer.produce(KAFKA_TOPIC_NAME, transaction)
    producer.flush()
