import logging
import os
import time
from pathlib import Path

import torch

from deployment.roles.haystack.files.custom_component import \
        TitleEmbeddingRetriever

from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever

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
from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
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

    p = pipelines.retriever_reader(parameters,
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            gpu_id = gpu_id,
            yaml_dir_prefix = yaml_dir_prefix)

    # Get all the retrievers used in the pipelines.
    retrievers = [p.get_node(name)
            for name in ["Retriever", "Retriever_bm25", "Retriever_title"] 
            if p.get_node(name)]

    # When there are multiple retrievers, the same document_store is attached
    # to all of them. Take the first one.
    document_store = retrievers[0].document_store

    preprocessing = parameters["preprocessing"]
    split_by = parameters["split_by"]
    split_length = int(parameters["split_length"])  # this is intended to convert numpy.int64 to int
    if preprocessing:
        preprocessor = pipelines.components.preprocessors.preprocessor(split_by, split_length)
    else:
        preprocessor = None

    # deleted indice for elastic search to make sure mappings are properly passed
    delete_indices(elasticsearch_hostname, elasticsearch_port, index="document_elasticsearch")
    delete_indices(elasticsearch_hostname, elasticsearch_port, index="label_elasticsearch")

    # Add evaluation data to Elasticsearch document store
    document_store.add_eval_data(
        evaluation_data.as_posix(),
        doc_index="document_elasticsearch",
        label_index="label_elasticsearch",
        preprocessor=preprocessor,
    )

    for retriever in retrievers:
        if type(retriever) in [DensePassageRetriever, TitleEmbeddingRetriever, 
                EmbeddingRetriever]:
            document_store.update_embeddings(retriever, index="document_elasticsearch")

    if parameters["retriever_type"] in ["title_bm25", "hot_reader"]:
        # used to make sure the p.run method returns enough candidates
        k_retriever = max(parameters["k_retriever"], parameters["k_title_retriever"])

    else:
        k_retriever = parameters["k_retriever"]

    k_reader_total = parameters["k_reader_total"]

    retriever_reader_eval_results = {}
    try:
        start = time.time()
        full_eval_retriever_reader(document_store=document_store,
                                   pipeline=p,
                                   k_retriever=k_retriever,
                                   k_reader_total=k_reader_total,
                                   label_index="label_elasticsearch")

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

    except Exception as e:
        logging.error(f"Could not run this config: {parameters}. Error {e}.")

    return retriever_reader_eval_results


def optimize(parameters, n_calls, result_file_path, gpu_id=-1,
             elasticsearch_hostname="localhost",
             elasticsearch_port=9200,
             yaml_dir_prefix="./output/pipelines/retriever_reader"):
    """ Returns a list of n_calls tuples [(x1, v1), ...] where the lists xi are
    the parameter values for each evaluation and the dictionaries vi are the run
    results. The parameter values for the successive runs are determined by the
    bayesian optimization method gp_minimize.
    """

    dimensions = create_dimensions_from_parameters(parameters)

    # TODO: optimize should return a generator rather than a list to be
    # consistent with the functions grid_search and tune_pipeline.
    results = []

    @use_named_args(dimensions=dimensions)
    def single_run_optimization(params):
        result = single_run(params, gpu_id = gpu_id,
                            elasticsearch_hostname = elasticsearch_hostname, 
                            elasticsearch_port = elasticsearch_port,
                            yaml_dir_prefix = yaml_dir_prefix)

        results.append((None, parameters, result))

        return 1 - result["reader_topk_accuracy_has_answer"]

    res = gp_minimize(
        single_run_optimization,
        dimensions,
        n_calls=n_calls,
        callback=LoggingCallback(n_calls),
        n_jobs=-1,
    )
    dump(res, result_file_path, store_objective=True)

    return results


def grid_search(parameters, mlflow_client, experiment_name, use_cache=False,
                result_file_path=Path("./output/results_reader.csv"), gpu_id=-1,
                elasticsearch_hostname="localhost", elasticsearch_port=9200,
                yaml_dir_prefix="./output/pipelines/retriever_reader"):
    """ Returns a generator of tuples [(id1, x1, v1), ...] where id1 is the run
    id, the lists xi are the parameter values for each evaluation and the
    dictionaries vi are the run results. The parameter values for each
    successive run are determined by a grid search method.
    """
    parameters_grid = list(ParameterGrid(param_grid=parameters))
    list_run_ids = create_run_ids(parameters_grid)
    list_past_run_names = get_list_past_run(mlflow_client, experiment_name)

    for idx, param in tqdm(
            zip(list_run_ids, parameters_grid),
            total=len(list_run_ids),
            desc="GridSearch",
            unit="config",
    ):
        enriched_param = add_extra_params(param)
        if (
                idx in list_past_run_names.keys() and use_cache
        ):  # run not done
            logging.info(
                f"Config {param} already done and found in mlflow. Not doing it again."
            )
            # Log again run with previous results
            previous_metrics = mlflow_client.get_run(list_past_run_names[idx]).data.metrics

            yield (idx, param, previous_metrics)

        else:  # run notalready done or USE_CACHE set to False or not set
            logging.info(f"Doing run with config : {param}")
            run_results = single_run(param, gpu_id = gpu_id,
                                     elasticsearch_hostname = elasticsearch_hostname,
                                     elasticsearch_port = elasticsearch_port,
                                     yaml_dir_prefix = yaml_dir_prefix)

            # For debugging purpose, we keep a copy of the results in a csv form
            save_results(result_file_path=result_file_path,
                         results_list={**run_results, **enriched_param})

            # update list of past experiments
            list_past_run_names = get_list_past_run(mlflow_client, experiment_name)

            yield (idx, param, run_results)




def tune_pipeline(
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

    experiment_name = parameter_tuning_options["experiment_name"]
    device, n_gpu = initialize_device_settings(use_cuda=True)
    GPU_AVAILABLE = 1 if device.type == "cuda" else 0

    if GPU_AVAILABLE:
        gpu_id = torch.cuda.current_device()
    else:
        gpu_id = -1

    launch_ES(elasticsearch_hostname, elasticsearch_port)
    client = MlflowClient()
    mlflow.set_experiment(experiment_name=experiment_name)

    if parameter_tuning_options["tuning_method"] == "optimization":
        runs = optimize(
            parameters=parameters,
            n_calls=parameter_tuning_options["optimization_ncalls"],
            result_file_path=Path("./output/optimize_result.z"),
            gpu_id=gpu_id,
            elasticsearch_hostname=elasticsearch_hostname,
            elasticsearch_port=elasticsearch_port,
            yaml_dir_prefix=yaml_dir_prefix)

    elif parameter_tuning_options["tuning_method"] == "grid_search":
        runs = grid_search(
            parameters=parameters,
            mlflow_client=client,
            experiment_name=parameter_tuning_options["experiment_name"],
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
    from src.evaluation.config.retriever_reader_eval_squad_config import \
        parameters, parameter_tuning_options

    yaml_dir_prefix = "./output/pipelines/retriever_reader"

    runs = tune_pipeline(
        parameters,
        parameter_tuning_options,
        elasticsearch_hostname=os.getenv("ELASTICSEARCH_HOSTNAME") or "localhost",
        elasticsearch_port=int((os.getenv("ELASTICSEARCH_PORT")) or 9200),
        yaml_dir_prefix = yaml_dir_prefix)

    for (run_id, params, results) in runs:
        clean_log()
        mlflow_log_run(params, results, idx = run_id,
                yaml_dir_prefix = yaml_dir_prefix)

