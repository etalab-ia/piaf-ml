import torch
import time
import os

import logging


logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./logs/run_log.log"),
        logging.StreamHandler()
    ]
)

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

from skopt import gp_minimize, dump
from skopt.plots import plot_objective
from skopt.utils import use_named_args

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever
from haystack.reader.transformers import TransformersReader
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.pipeline import ExtractiveQAPipeline

from farm.utils import initialize_device_settings
from sklearn.model_selection import ParameterGrid

from src.evaluation.config.retriever_reader_eval_squad_config import parameters
from src.evaluation.utils.TitleEmbeddingRetriever import TitleEmbeddingRetriever
from src.evaluation.utils.elasticsearch_management import launch_ES, delete_indices, prepare_mapping
from src.evaluation.utils.mlflow_management import prepare_mlflow_server, add_extra_params, create_run_ids, get_list_past_run
from src.evaluation.utils.utils_eval import eval_retriever_reader, save_results
from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
from src.evaluation.utils.custom_pipelines import TitleBM25QAPipeline
from src.evaluation.utils.utils_optimizer import tqdm_skopt, create_dimensions_from_parameters

load_dotenv()
prepare_mlflow_server()

GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    n_gpu = torch.cuda.current_device()
else:
    n_gpu = -1


def single_run(idx=None, **kwargs):
    """
        Queries ES max_k - min_k times, saving at each step the results in a list. At the end plots the line
        showing the results obtained. For now we can only vary k.
        :param min_k: Minimum retriever-k to test
        :param max_k: Maximum retriever-k to test
        :param weighted_precision: Whether to take into account the position of the retrieved result in the accuracy computation
        :return:
        """
    with mlflow.start_run(run_name=idx) as run:
        mlflow.log_params(kwargs)
        # col names
        evaluation_data = Path(kwargs["squad_dataset"])
        retriever_type = kwargs["retriever_type"]
        k_retriever = kwargs["k_retriever"]
        k_title_retriever = kwargs["k_title_retriever"]
        k_reader_per_candidate = kwargs["k_reader_per_candidate"]
        k_reader_total = kwargs["k_reader_total"]
        preprocessing = kwargs["preprocessing"]
        reader_model_version = kwargs["reader_model_version"]
        retriever_model_version = kwargs["retriever_model_version"]
        split_by = kwargs["split_by"]
        split_length = int(kwargs["split_length"]) #this is intended to convert numpy.int64 to int
        title_boosting_factor = kwargs["boosting"]

        doc_index = "document_xp"
        label_index = "label_xp"

        # deleted indice for elastic search to make sure mappings are properly passed
        delete_indices(index=doc_index)
        delete_indices(index=label_index)


        prepare_mapping(mapping=SQUAD_MAPPING, title_boosting_factor=title_boosting_factor, embedding_dimension=768)

        if preprocessing:
            preprocessor = PreProcessor(
                clean_empty_lines=False,
                clean_whitespace=False,
                clean_header_footer=False,
                split_by=split_by,
                split_length=split_length,
                split_overlap=0,  # this must be set to 0 at the data of writting this: 22 01 2021
                split_respect_sentence_boundary=False  # the support for this will soon be removed : 29 01 2021
            )
        else:
            preprocessor = None

        reader = TransformersReader(model_name_or_path="etalab-ia/camembert-base-squadFR-fquad-piaf",
                                    tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf",
                                    model_version=reader_model_version,
                                    use_gpu=gpu_id, top_k_per_candidate=k_reader_per_candidate)

        if retriever_type == 'bm25':

            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                        search_fields=["name", "text"],
                                                        create_index=False, embedding_field="emb",
                                                        scheme="",
                                                        embedding_dim=768, excluded_meta_data=["emb"], similarity='cosine',
                                                        custom_mapping=SQUAD_MAPPING)
            retriever = ElasticsearchRetriever(document_store=document_store)
            p = ExtractiveQAPipeline(reader=reader, retriever=retriever)

        elif retriever_type == "sbert":
            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                        search_fields=["name", "text"],
                                                        create_index=False, embedding_field="emb",
                                                        embedding_dim=768, excluded_meta_data=["emb"], similarity='cosine',
                                                        custom_mapping=SQUAD_MAPPING)
            retriever = EmbeddingRetriever(document_store=document_store,
                                           embedding_model="distilbert-base-multilingual-cased",
                                           model_version=retriever_model_version,
                                           use_gpu=GPU_AVAILABLE, model_format="transformers",
                                           pooling_strategy="reduce_max",
                                           emb_extraction_layer=-1)
            p = ExtractiveQAPipeline(reader=reader, retriever=retriever)

        elif retriever_type == "title_bm25":
            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                        search_fields=["name", "text"],
                                                        create_index=False, embedding_field="emb",
                                                        embedding_dim=768, excluded_meta_data=["emb"], similarity='cosine',
                                                        custom_mapping=SQUAD_MAPPING)
            retriever = TitleEmbeddingRetriever(document_store=document_store,
                                                embedding_model="distilbert-base-multilingual-cased",
                                                model_version=retriever_model_version,
                                                use_gpu=GPU_AVAILABLE, model_format="transformers",
                                                pooling_strategy="reduce_max",
                                                emb_extraction_layer=-1)
            retriever_bm25 = ElasticsearchRetriever(document_store=document_store)

            p = TitleBM25QAPipeline(reader=reader, retriever_title=retriever, retriever_bm25=retriever_bm25,
                                    k_title_retriever=k_title_retriever
                                    , k_bm25_retriever=k_retriever)

            # used to make sure the p.run method returns enough candidates
            k_retriever = max(k_retriever, k_title_retriever)

        elif retriever_type == "title":
            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                        search_fields=["name", "text"],
                                                        create_index=False, embedding_field="emb",
                                                        embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                        custom_mapping=SQUAD_MAPPING)
            retriever = TitleEmbeddingRetriever(document_store=document_store,
                                                embedding_model="distilbert-base-multilingual-cased",
                                                model_version=retriever_model_version,
                                                use_gpu=GPU_AVAILABLE, model_format="sentence_transformers",
                                                pooling_strategy="reduce_max",
                                                emb_extraction_layer=-1)
            p = ExtractiveQAPipeline(reader=reader, retriever=retriever)


        else:
            logging.error(f"You chose {retriever_type}. Choose one from bm25, sbert, dpr or title_bm25")
            raise Exception(f"You chose {retriever_type}. Choose one from bm25, sbert, dpr or title_bm25")
        # Add evaluation data to Elasticsearch document store
        # We first delete the custom tutorial indices to not have duplicate elements
        # make sure these indices do not collide with existing ones, the indices will be wiped clean before data is inserted

        document_store.add_eval_data(evaluation_data.as_posix(), doc_index=doc_index, label_index=label_index,
                                     preprocessor=preprocessor)

        if retriever_type in ["sbert", "dpr", "title_bm25", "title"]:
            document_store.update_embeddings(retriever, index=doc_index)

        retriever_eval_results = {}
        try:
            start = time.time()
            retriever_eval_results = eval_retriever_reader(document_store=document_store, pipeline=p,
                                                           k_retriever=k_retriever, k_reader_total=k_reader_total,
                                                           label_index=label_index)
            end = time.time()
            time_per_label = (end - start) / document_store.get_label_count(index=label_index)

            print("Reader Accuracy:", retriever_eval_results["reader_topk_accuracy_has_answer"])
            print("reader_topk_f1:", retriever_eval_results["reader_topk_f1"])

            retriever_eval_results.update({"time_per_label": time_per_label})
        except Exception as e:
            logging.error(f"Could not run this config: {kwargs}. Error {e}.")

        mlflow.log_metrics({k: v for k, v in retriever_eval_results.items() if v is not None})

    return retriever_eval_results


if __name__ == '__main__':
    result_file_path = Path("./results/results_reader.csv")
    optimize_result_file_path = Path("./results/optimize_result.z")
    parameters_grid = list(ParameterGrid(param_grid=parameters))
    experiment_name = parameters["experiment_name"][0]

    device, n_gpu = initialize_device_settings(use_cuda=True)
    GPU_AVAILABLE = 1 if device.type == "cuda" else 0

    if GPU_AVAILABLE:
        gpu_id = torch.cuda.current_device()
    else:
        gpu_id = -1

    all_results = []

    launch_ES()
    client = MlflowClient()
    mlflow.set_experiment(experiment_name=experiment_name)

    if os.getenv('USE_OPTIMIZATION') == 'True':
        dimensions = create_dimensions_from_parameters(parameters)

        @use_named_args(dimensions=dimensions)
        def single_run_optimization(**kwargs):
            return 1 - single_run(**kwargs)["reader_topk_accuracy_has_answer"]
        n_calls = os.getenv('OPTIMIZATION_NCALLS')
        res = gp_minimize(single_run_optimization, dimensions, n_calls=n_calls, n_jobs=1)
        #callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")],
        dump(res, optimize_result_file_path, store_objective=True)

    else:
        parameters_grid = list(ParameterGrid(param_grid=parameters))
        list_run_ids = create_run_ids(parameters_grid)
        list_past_run_names = get_list_past_run(client, experiment_name)

        for idx, param in tqdm(zip(list_run_ids, parameters_grid), total=len(list_run_ids), desc="GridSearch",
                               unit="config"):
            add_extra_params(param)
            if idx in list_past_run_names.keys() and os.getenv("USE_CACHE") == "True":  # run not done
                print('config already done')
                # Log again run with previous results
                previous_metrics = client.get_run(list_past_run_names[idx]).data.metrics
                with mlflow.start_run(run_name=idx) as run:
                    mlflow.log_params(param)
                    mlflow.log_metrics(previous_metrics)

            else:  # run notalready done or USE_CACHE set to False or not set
                logging.info(f"Doing run with config : {param}")
                #try:
                run_results = single_run(idx=idx, **param)
                run_results.update(param)
                save_results(result_file_path=result_file_path, results_list=run_results)
                list_past_run_names = get_list_past_run(client, experiment_name) # update list of past experiments
                """
                except Exception as e:
                    logging.error(f"Could not run this config: {param}. Error {e}.")
                    continue"""


