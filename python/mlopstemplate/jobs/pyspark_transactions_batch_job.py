import hopsworks

from mlopstemplate.jobs.jobutil import JobUtils

project = hopsworks.login()

JobUtils.run_job(project,
                 "../pipelines/pyspark_pipelines/transaction.py",
                 "transaction_pipeline",
                 max_executors=5)
