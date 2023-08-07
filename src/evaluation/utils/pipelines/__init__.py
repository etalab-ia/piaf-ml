"""
This root module contains functions that build pipelines given a pipeline
parameter dictionnary and a few other arguments. They are responsible for
calling the right pipeline constructor function from the submodule
`custom_pipelines` from the values in the parameter dict.

There are two submodules: `custom_pipelines` and `components`.

The `custom_pipelines` submodule contains functions that build pipelines.

The `components` submodule contains functions that create the custom
components used in the pipelines defined in the `custom_pipelines` submodule.
"""


import hashlib
from haystack.pipeline import Pipeline
import json
from pathlib import Path

import src.evaluation.utils.pipelines.custom_pipelines as custom_pipelines

def retriever(
        parameters,
        elasticsearch_hostname,
        elasticsearch_port,
        gpu_id = -1,
        yaml_dir_prefix = "./output/pipelines/"):

    retriever_type = parameters["retriever_type"]

    if retriever_type == "bm25":
        pipeline =  custom_pipelines.retriever_bm25(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            k = parameters["k"])

    elif retriever_type == "sbert":
        pipeline = custom_pipelines.retriever_sbert(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            gpu_available = gpu_id >= 0)

    elif retriever_type == "google":
        pipeline = custom_pipelines.retriever_google(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            google_retriever_website = parameters["google_retriever_website"])

    elif retriever_type == "epitca":
        pipeline = custom_pipelines.retriever_epitca(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"])

    else:
        raise Exception(
            f"You chose {retriever_type}. Choose one from bm25, sbert, google or dpr"
        )

    # Save the pipeline to yaml and load it back from yaml to make sure the 
    # pipeline being evaluated is build the same way as it is in prod.
    pipeline = pipeline_to_yaml_and_back(pipeline, parameters,
            prefix = Path(yaml_dir_prefix))

    return pipeline


def retriever_reader(
        parameters,
        elasticsearch_hostname = "localhost",
        elasticsearch_port = 9200,
        gpu_id = -1,
        yaml_dir_prefix = "./output/pipelines",
        ):

    # Gather parameters
    retriever_type = parameters["retriever_type"]

    if retriever_type == "bm25":
        pipeline = custom_pipelines.retriever_reader_bm25(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_retriever = parameters["k_retriever"],
            k_reader_total = parameters["k_reader_total"],
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "sbert":
        pipeline = custom_pipelines.retriever_reader_sbert(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "dpr":
        pipeline = custom_pipelines.retriever_reader_dpr(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            dpr_model_version = parameters["dpr_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "title_bm25":
        pipeline = custom_pipelines.retriever_reader_title_bm25(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"],
            k_title_retriever = parameters["k_title_retriever"],
            k_bm25_retriever = parameters["k_retriever"])

    elif retriever_type == "title":
        pipeline = custom_pipelines.retriever_reader_title(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "hot_reader":
        pipeline = custom_pipelines.hottest_reader_pipeline(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"],
            k_title_retriever = parameters["k_title_retriever"],
            k_bm25_retriever = parameters["k_retriever"],
            threshold_score = parameters["threshold_score"])

    else:
        logging.error(
            f"You chose {retriever_type}. Choose one from bm25, sbert, dpr, title_bm25 or title."
        )
        raise Exception(f"Wrong retriever type for {retriever_type}.")

    # Save the pipeline to yaml and load it back from yaml to make sure the 
    # pipeline being evaluated is build the same way as it is in prod.
    pipeline = pipeline_to_yaml_and_back(pipeline, parameters,
            prefix = Path(yaml_dir_prefix))

    return pipeline

def client(
        client,
        parameters,
        elasticsearch_hostname = "localhost",
        elasticsearch_port = 9200,
        gpu_id = -1,
        yaml_dir_prefix = "./output/pipelines",
        is_eval = False,
        ):

    if client == "cnil":
        pipeline = custom_pipelines.cnil(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_total = parameters["k_reader_total"],
            k_reader_per_candidate = parameters["k_reader_per_candidate"],
            k_title_retriever = parameters["k_title_retriever"],
            k_bm25_retriever = parameters["k_bm25_retriever"],
            k_label_retriever = parameters["k_label_retriever"],
            ks_retriever = parameters["ks_retriever"],
            weight_when_document_found = parameters["weight_when_document_found"],
            threshold_score = parameters["threshold_score"],
            context_window_size = parameters["context_window_size"],
            is_eval = is_eval)

    else:
        raise Exception(f"Unknown client {client}")

    # Save the pipeline to yaml and load it back from yaml to make sure the 
    # pipeline being evaluated is build the same way as it is in prod.
    pipeline = pipeline_to_yaml_and_back(pipeline, parameters,
            prefix = Path(yaml_dir_prefix))

    return pipeline

def pipeline_to_yaml_and_back(pipeline, parameters, prefix = "./output/pipelines/"):
    dirname = pipeline_dirpath(parameters, prefix)
    dirname.mkdir(parents = True, exist_ok = True)

    yaml_path = dirname / "pipelines.yaml"
    pipeline.save_to_yaml(yaml_path, return_defaults = True)

    params_path = dirname / "params.py"
    with open(params_path, "w") as f:
        f.write(repr(parameters))

    # Turn off overwriting with env variable to avoid accidentally constructing
    # a different pipeline that the one defined in the yaml.
    return Pipeline.load_from_yaml(yaml_path, overwrite_with_env_variables = False)

def pipeline_dirpath(parameters, prefix = "./output/pipelines/"):
    params_hash = hashlib.sha1(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    path = params_hash
    return Path(prefix) / path
