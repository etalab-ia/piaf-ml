#!/usr/bin/env python3

from elasticsearch import Elasticsearch
import os
import sys
import time

from src.evaluation.retriever_reader.retriever_reader_eval_squad import \
    tune_pipeline
from src.evaluation.config.retriever_reader_eval_squad_config import \
    parameters, parameter_tuning_options

test_dataset = os.getenv("DATA_DIR") + "/non-regression-tests/eval_dataset.json"
elasticsearch_hostname = os.getenv("ELASTICSEARCH_HOSTNAME")
elasticsearch_port = int(os.getenv("ELASTICSEARCH_PORT"))

parameters["squad_dataset"] = [test_dataset]
parameter_tuning_options["experiment_name"] = "non-regression-tests"

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

tune_pipeline(
    parameters,
    parameter_tuning_options,
    elasticsearch_hostname = elasticsearch_hostname,
    elasticsearch_port = elasticsearch_port)
