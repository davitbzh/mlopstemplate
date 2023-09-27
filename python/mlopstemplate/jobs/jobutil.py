import subprocess, os


class JobUtils:

    @staticmethod
    def run_job(project, driver_code_path, job_name, args=None, max_executors=2, min_executors=1,
                job_type="PYSPARK", await_termination=True, download_logs=True):

        dataset_api = project.get_dataset_api()

        print("upload driver job")
        uploaded_driver_code = dataset_api.upload(driver_code_path, "Resources", overwrite=True)

        jobs_api = project.get_jobs_api()
        spark_config = jobs_api.get_configuration(job_type)
        spark_config['appPath'] = uploaded_driver_code
        # spark_config['spark.yarn.dist.pyFiles'] = "hdfs:///Projects/{}/{}".format(project.name, uploaded_test_code)
        spark_config['spark.dynamicAllocation.maxExecutors'] = max_executors
        spark_config['spark.dynamicAllocation.minExecutors'] = min_executors
        print("create job")
        job = jobs_api.create_job(job_name, spark_config)
        print("run job")
        execution = job.run(await_termination=await_termination, args=args)

        out = None
        err = None
        if download_logs is True:
            out, err = execution.download_logs()

        out_log = "NO LOG AGGREGATED"
        if out is not None:
            out_log = open(out, "r").read()

        err_log = "NO LOG AGGREGATED"
        if err is not None:
            err_log = open(err, "r").read()

        assert execution.success, "STDOUT: {} /n/n STDERR: {}".format(out_log, err_log)
        return execution

    @staticmethod
    def build_flink_jar(hopsworks_version):
        tests_dir_path = os.getcwd()
        try:
            def run_cmd(command):
                return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()

            java_integrations_dir_path = "hopsworks-tutorials/integrations/java"
            relative_flink_dir_path = "hopsworks-tutorials/integrations/java/flink"
            integrations_dir = os.path.join(os.getcwd(), java_integrations_dir_path)
            flink_dir = os.path.join(os.getcwd(), relative_flink_dir_path)
            # Only first test will build the flink jar
            if not os.path.exists(integrations_dir):
                print(run_cmd("git clone https://github.com/logicalclocks/hopsworks-tutorials.git"))
                os.chdir(integrations_dir)
                print(run_cmd("mvn versions:set -DnewVersion={}".format(hopsworks_version)))
                os.chdir(flink_dir)
                print(run_cmd("mvn clean install"))
        finally:
            os.chdir(tests_dir_path)
        return os.path.join(flink_dir, "target", "flink-{}.jar".format(hopsworks_version))
