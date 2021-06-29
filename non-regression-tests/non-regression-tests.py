#!/usr/bin/env python3

from elasticsearch import Elasticsearch
import os
import sys
import time

from src.evaluation.retriever_reader.retriever_reader_eval_squad import \
    tune_pipeline
from src.evaluation.config.retriever_reader_eval_squad_config import \
    parameters, parameter_tuning_options

# TODO GPUBRO use default retriever type. This is just for testing
parameters["retriever_type"] = ["bm25"] # Can be bm25, sbert, dpr, title or title_bm25
parameters["squad_dataset"] = ["/usr/local/share/non-regression-tests/fquad_eval.json"]
parameter_tuning_options["experiment_name"] = "non-regression-test"

elasticsearch_hostname = os.getenv("ELASTICSEARCH_HOSTNAME") or "localhost"
elasticsearch_port = int(os.getenv("ELASTICSEARCH_PORT")) or 9200

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
