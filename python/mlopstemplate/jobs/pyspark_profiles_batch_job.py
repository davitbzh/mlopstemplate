import hopsworks

from jobutil import JobUtils

project = hopsworks.login()

JobUtils.run_job(project,
                 "../pipelines/pyspark_pipelines/profile.py",
                 "profile_pipeline",
                 max_executors=5)
