#!/usr/bin/env python3

from elasticsearch import Elasticsearch
import os
import sys
import time

import config
from src.evaluation.retriever_reader.retriever_reader_eval_squad import \
    tune_pipeline
from src.evaluation.utils.logging_management import clean_log
from src.evaluation.utils.mlflow_management import mlflow_log_run

elasticsearch_hostname = os.getenv("ELASTICSEARCH_HOSTNAME")
elasticsearch_port = int(os.getenv("ELASTICSEARCH_PORT"))

# Wait for elasticsearch
timeout = 60 #seconds
start_wait = time.time()

es = Elasticsearch([f"http://{elasticsearch_hostname}:{elasticsearch_port}/"], verify_certs=True)
while not es.ping():
    print(f"Waiting for Elasticsearch to be ready at " \
        + f"http://{elasticsearch_hostname}:{elasticsearch_port}")
    sys.stdout.flush()
    time.sleep(1)
    if time.time() - start_wait > timeout:
        print("Timeout waiting for Elasticsearch at " \
            + "http://{elasticsearch_hostname}:{elasticsearch_port}. " \
            + "Elasticsearch not found.")



# For each test in config.test, log all runs and set a tag if the metrics are
# not satisfying.
for parameters, parameter_tuning_options, pass_criteria in config.tests:
    runs = tune_pipeline(
        parameters,
        parameter_tuning_options,
        elasticsearch_hostname = elasticsearch_hostname,
        elasticsearch_port = elasticsearch_port)

    for (run_id, params, results) in runs:
        passed = all(pass_criteria[k](results[k]) for k in pass_criteria.keys())

        clean_log()
        mlflow_log_run(params, results, idx=run_id, pass_criteria = passed)



