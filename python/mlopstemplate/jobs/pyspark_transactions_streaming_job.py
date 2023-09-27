import hopsworks

from mlopstemplate.jobs.jobutil import JobUtils

project = hopsworks.login()

JobUtils.run_job(project,
                 "../pipelines/pyspark_pipelines/transactions_streaming.py",
                 "transaction_streaming_pipeline",
                 max_executors=2)
