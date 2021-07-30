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

from pathlib import Path
from haystack.pipeline import Pipeline

import src.evaluation.utils.pipelines.custom_pipelines as custom_pipelines

def retriever(
        parameters,
        elasticsearch_hostname,
        elasticsearch_port,
        gpu_id = -1):

    retriever_type = parameters["retriever_type"]

    if retriever_type == "bm25":
        return custom_pipelines.retriever_bm25(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"])

    elif retriever_type == "sbert":
        return custom_pipelines.retriever_sbert(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            gpu_available = gpu_id >= 0)

    elif retriever_type == "google":
        return custom_pipelines.retriever_google(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            google_retriever_website = parameters["google_retriever_website"])

    elif retriever_type == "epitca":
        return custom_pipelines.retriever_epitca(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"])

    else:
        raise Exception(
            f"You chose {retriever_type}. Choose one from bm25, sbert, google or dpr"
        )


def retriever_reader(
        parameters,
        gpu_id = -1,
        elasticsearch_hostname = "localhost",
        elasticsearch_port = 9200,
        ):

    # Gather parameters
    retriever_type = parameters["retriever_type"]

    if retriever_type == "bm25":
        return custom_pipelines.retriever_reader_bm25(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "sbert":
        return custom_pipelines.retriever_reader_sbert(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "dpr":
        return custom_pipelines.retriever_reader_dpr(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            dpr_model_version = parameters["dpr_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "title_bm25":
        return custom_pipelines.retriever_reader_title_bm25(
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
        return custom_pipelines.retriever_reader_title(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            title_boosting_factor = parameters["boosting"],
            retriever_model_version = parameters["retriever_model_version"],
            reader_model_version = parameters["reader_model_version"],
            gpu_id = gpu_id,
            k_reader_per_candidate = parameters["k_reader_per_candidate"])

    elif retriever_type == "hot_reader":
        return custom_pipelines.hottest_reader_pipeline(
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
