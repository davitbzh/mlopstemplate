import hopsworks

from mlopstemplate.jobs.jobutil import JobUtils

project = hopsworks.login()

JobUtils.run_job(project,
                 "../synthetic_data/transactions_stream_simulator.py",
                 "transactions_simulator",
                 max_executors=5)
