import logging
import os
import time
from pathlib import Path
import importlib

import torch

from deployment.roles.haystack.files.custom_component import \
        TitleEmbeddingRetriever

from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever
from haystack.retriever.base import BaseRetriever

from src.evaluation.utils.logging_management import clean_log
import mlflow
from dotenv import load_dotenv
from farm.utils import BaseMLLogger, initialize_device_settings
from mlflow.tracking import MlflowClient
import pprint
from sklearn.model_selection import ParameterGrid
from skopt import dump, gp_minimize
from skopt.utils import use_named_args
import sys
from tqdm import tqdm

from src.evaluation.utils.utils_eval import save_results, \
    full_eval_retriever_reader
from src.evaluation.utils.elasticsearch_management import delete_indices, \
    launch_ES, prepare_mapping
from src.evaluation.utils.mlflow_management import add_extra_params, \
    create_run_ids, get_list_past_run, prepare_mlflow_server, mlflow_log_run
from src.evaluation.utils.utils_optimizer import LoggingCallback, \
    create_dimensions_from_parameters

import src.evaluation.utils.pipelines as pipelines

BaseMLLogger.disable_logging = True
load_dotenv()
prepare_mlflow_server()

GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    n_gpu = torch.cuda.current_device()
else:
    n_gpu = -1


def single_run(
        client,
        parameters,
        idx=None,
        gpu_id = -1,
        elasticsearch_hostname = "localhost",
        elasticsearch_port = 9200,
        yaml_dir_prefix = "./output/pipelines/retriever_reader",
        ):
    """
    Perform one run of the pipeline under testing with the parameters given in the config file. The results are
    saved in the mlflow instance.
    """

    evaluation_data = Path(parameters["squad_dataset"])

    logging.info("Building pipeline.")
    p = pipelines.client(
            client = client,
            parameters = parameters,
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            gpu_id = gpu_id,
            yaml_dir_prefix = yaml_dir_prefix,
            is_eval = True)
    logging.info("Pipeline built.")

    # Get all the retrievers used in the pipelines.
    retrievers = [p.get_node(node) for node in p.graph.nodes
            if isinstance(p.get_node(node), BaseRetriever)]

    # When there are multiple retrievers, the same document_store is attached
    # to all of them. Take the first one.
    document_store = retrievers[0].document_store

    # deleted indice for elastic search to make sure mappings are properly passed
    logging.info("Deleting elasticsearch indices.")
    delete_indices(elasticsearch_hostname, elasticsearch_port, index="document_elasticsearch")
    delete_indices(elasticsearch_hostname, elasticsearch_port, index="label_elasticsearch")

    logging.info("Adding evaluation data to Elasticsearch.")
    # Add evaluation data to Elasticsearch document store
    document_store.add_eval_data(
        evaluation_data.as_posix(),
        doc_index="document_elasticsearch",
        label_index="label_elasticsearch"
    )

    for retriever in retrievers:
        if type(retriever) in [DensePassageRetriever, TitleEmbeddingRetriever, 
                EmbeddingRetriever]:
            document_store.update_embeddings(retriever, index="document_elasticsearch")

    # used to make sure the p.run method returns enough candidates
    k_retriever = max(
        [parameters["k_bm25_retriever"]] if "k_bm25_retriever" in parameters else [] \
        + [parameters["k_title_retriever"]] if "k_title_retriever" in parameters else [] \
        + [parameters["k_retriever"]] if "k_retriever" in parameters else [])

    k_reader_total = parameters["k_reader_total"]

    retriever_reader_eval_results = {}

    start = time.time()
    logging.info("Evaluating the pipeline.")
    full_eval_retriever_reader(document_store=document_store,
                               pipeline=p,
                               k_retriever=k_retriever,
                               k_reader_total=k_reader_total,
                               label_index="label_elasticsearch")
    logging.info("Evaluation done.")

    eval_retriever = p.get_node("EvalRetriever")
    eval_reader = p.get_node("EvalReader")

    retriever_reader_eval_results.update(eval_retriever.get_metrics())
    retriever_reader_eval_results.update(eval_reader.get_metrics())

    end = time.time()

    logging.info(f"Retriever Recall: {retriever_reader_eval_results['recall']}")
    logging.info(f"Retriever Mean Avg Precision: {retriever_reader_eval_results['map']}")
    logging.info(f"Retriever Mean Reciprocal Rank: {retriever_reader_eval_results['mrr']}")
    logging.info(f"Reader Accuracy: {retriever_reader_eval_results['reader_topk_accuracy_has_answer']}")
    logging.info(f"reader_topk_f1: {retriever_reader_eval_results['reader_topk_f1']}")

    # Log time per label in metrics
    time_per_label = (end - start) / document_store.get_label_count(
        index="label_elasticsearch"
    )
    retriever_reader_eval_results.update({"time_per_label": time_per_label})

    return retriever_reader_eval_results



def grid_search(client, parameters, use_cache=False,
                result_file_path=Path("./output/results_reader.csv"), gpu_id=-1,
                elasticsearch_hostname="localhost", elasticsearch_port=9200,
                yaml_dir_prefix="./output/pipelines/retriever_reader"):
    """ Returns a generator of tuples [(id1, x1, v1), ...] where id1 is the run
    id, the lists xi are the parameter values for each evaluation and the
    dictionaries vi are the run results. The parameter values for each
    successive run are determined by a grid search method.
    """
    logging.info("Preparing parameter grid.")
    parameters_grid = list(ParameterGrid(param_grid=parameters))
    logging.info("Creating run ids.")
    list_run_ids = create_run_ids(parameters_grid)
    logging.info("Retrieving the list of previous runs.")

    logging.info("Starting grid search")
    for idx, param in tqdm(
            zip(list_run_ids, parameters_grid),
            total=len(list_run_ids),
            desc="GridSearch",
            unit="config"):

        logging.info(f"Doing run with config : {param}")
        run_results = single_run(client = client,
                                 parameters = param,
                                 gpu_id = gpu_id,
                                 elasticsearch_hostname = elasticsearch_hostname,
                                 elasticsearch_port = elasticsearch_port,
                                 yaml_dir_prefix = yaml_dir_prefix)
        logging.info(f"Run finished.")

        yield (idx, param, run_results)




def tune_pipeline(
        client,
        parameters,
        parameter_tuning_options,
        elasticsearch_hostname,
        elasticsearch_port,
        yaml_dir_prefix):
    """
    Run the parameter tuning method for the whole pipeline based on the
    parameters.

    Returns a generator of tuples (x1, v1), ... where the dicts xi are the
    parameters and v1 are the results for each run.
    """

    logging.info("\n---- Parameters ----\n" + pprint.pformat(parameters))
    logging.info("\n---- Parameter tuning options ----\n" + pprint.pformat(parameter_tuning_options))

    device, n_gpu = initialize_device_settings(use_cuda=True)
    GPU_AVAILABLE = 1 if device.type == "cuda" else 0

    if GPU_AVAILABLE:
        gpu_id = torch.cuda.current_device()
    else:
        gpu_id = -1

    logging.info("Launching Elasticsearch.")
    launch_ES(elasticsearch_hostname, elasticsearch_port)

    if parameter_tuning_options["tuning_method"] == "grid_search":
        runs = grid_search(
            client = client,
            parameters=parameters,
            use_cache=parameter_tuning_options["use_cache"],
            gpu_id=gpu_id,
            result_file_path=Path("./output/results_reader.csv"),
            elasticsearch_hostname=elasticsearch_hostname,
            elasticsearch_port=elasticsearch_port,
            yaml_dir_prefix=yaml_dir_prefix)

    else:
        print("Unknown parameter tuning method: ",
              parameter_tuning_options["tuning_method"],
              file=sys.stderr)
        exit(1)

    return runs





if __name__ == "__main__":

    for subdir in Path("clients/").iterdir():

        if not (subdir / "experiment" / "config.py").exists():
            continue

        client = subdir.name

        config = importlib.import_module(f"clients.{client}.experiment.config")

        logging.info(f"Running tuning pipeline experiment for client {client}")

        yaml_dir_prefix = Path("output/") / client / "pipelines"

        runs = tune_pipeline(
            client,
            config.parameters,
            config.parameter_tuning_options,
            elasticsearch_hostname=os.getenv("ELASTICSEARCH_HOSTNAME") or "localhost",
            elasticsearch_port=int((os.getenv("ELASTICSEARCH_PORT")) or 9200),
            yaml_dir_prefix = yaml_dir_prefix)

        logging.info("Preparing mlflow client.")

        for (run_id, params, results) in runs:
            clean_log()
            logging.info("Logging run result to mlflow.")
            mlflow_log_run(
                   experiment_name = config.parameter_tuning_options["experiment_name"],
                   params = params, 
                   results = results, 
                   idx = run_id,
                   yaml_dir_prefix = yaml_dir_prefix)
            logging.info("Logging to mlflow finished.")

