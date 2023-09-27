import hopsworks

from mlopstemplate.jobs.jobutil import JobUtils

project = hopsworks.login()

JobUtils.run_job(project,
                 "../features/synthetic/transactions_stream_simulator.py",
                 "transactions_simulator",
                 max_executors=5)
