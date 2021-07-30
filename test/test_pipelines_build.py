import pytest

import src.evaluation.utils.pipelines as pipelines

def test_retriever_build():
    parameters = {
        "k": 1,
        "retriever_type": "bm25",
        "retriever_model_version": "1a01b38498875d45f69b2a6721bf6fe87425da39",
        "google_retriever_website": 'service-public.fr',
        "squad_dataset": "./test/samples/squad/tiny.json",
        # Path to the Epitca performance file or None. Needed when using the
        # retriever_type 'epitca'.
        #"epitca_perf_file": "./clients/cnil/knowledge_base/raw_and_preparation/epitca_perf_V2.json",
        "epitca_perf_file": None,
        "filter_level": None,
        "boosting": 1,
        "preprocessing": False,
        "split_by": "word",  # Can be "word", "sentence", or "passage"
        "split_length": 200,
        "split_respect_sentence_boundary": True,
    }

    parameters["retriever_type"] = "bm25"
    pipelines.retriever(parameters, "localhost", "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever")

    parameters["retriever_type"] = "sbert"
    pipelines.retriever(parameters, "localhost", "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever")

    parameters["retriever_type"] = "google"
    pipelines.retriever(parameters, "localhost", "9200", gpu_id = -1,
           yaml_dir_prefix = "output/test/pipelines/retriever")

    parameters["retriever_type"] = "epitca"
    pipelines.retriever(parameters, "localhost", "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever")



def test_retriever_reader_build():
    parameters = {
        "k_retriever": 3,
        "k_title_retriever" : 1, # must be present, but only used when retriever_type == title_bm25
        "k_reader_per_candidate": 20,
        "k_reader_total": 5,
        "threshold_score": 1.00,# must be present, but only used when retriever_type == hot_reader
        "reader_model_version": "053b085d851196110d7a83d8e0f077d0a18470be",
        "retriever_model_version": "1a01b38498875d45f69b2a6721bf6fe87425da39",
        "dpr_model_version": "v1.0",
        "retriever_type": "bm25", # Can be bm25, sbert, dpr, title or title_bm25
        "squad_dataset": "./test/samples/squad/tiny.json",
        "filter_level": None,
        "preprocessing": False,
        "boosting" : 1, #default to 1
        "split_by": "word",  # Can be "word", "sentence", or "passage"
        "split_length": 1000,
    }

    parameters["retriever_type"] = "bm25"
    pipelines.retriever_reader(parameters, elasticsearch_hostname = "localhost",
            elasticsearch_port = "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever_reader")

    parameters["retriever_type"] = "sbert"
    pipelines.retriever_reader(parameters, elasticsearch_hostname = "localhost",
            elasticsearch_port = "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever_reader")

    parameters["retriever_type"] = "dpr"
    pipelines.retriever_reader(parameters, elasticsearch_hostname = "localhost",
            elasticsearch_port = "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever_reader")

    parameters["retriever_type"] = "title"
    pipelines.retriever_reader(parameters, elasticsearch_hostname = "localhost",
            elasticsearch_port = "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever_reader")

    parameters["retriever_type"] = "title_bm25"
    pipelines.retriever_reader(parameters, elasticsearch_hostname = "localhost",
            elasticsearch_port = "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever_reader")

    parameters["retriever_type"] = "hot_reader"
    pipelines.retriever_reader(parameters, elasticsearch_hostname = "localhost",
            elasticsearch_port = "9200", gpu_id = -1,
            yaml_dir_prefix = "output/test/pipelines/retriever_reader")
