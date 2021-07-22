import os
import hashlib
import socket
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from pprint import pprint

from deployment.roles.haystack.files.custom_component import \
        TitleEmbeddingRetriever

from farm.utils import initialize_device_settings
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.pipeline import Pipeline
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
from src.evaluation.config.retriever_eval_squad_config import parameters
from src.evaluation.utils.elasticsearch_management import (delete_indices,
                                                           launch_ES,
                                                           prepare_mapping)
from src.evaluation.utils import epitca_retriever
from src.evaluation.utils.utils_eval import eval_retriever, save_results

import src.evaluation.utils.pipelines as pipelines

device, n_gpu = initialize_device_settings(use_cuda=True)
GPU_AVAILABLE = 1 if device.type == "cuda" else 0

if GPU_AVAILABLE:
    gpu_id = torch.cuda.current_device()
else:
    gpu_id = -1

def single_run(parameters, elasticsearch_hostname, elasticsearch_port):
    """
    Runs a grid search config 

    :param parameters: A dict with diverse config options 
    :return: A dict with the results obtained running the experiment with these parameters
    """
    # col names
    evaluation_data = Path(parameters["squad_dataset"])
    retriever_type = parameters["retriever_type"]
    k = parameters["k"]
    epitca_perf_file = parameters["epitca_perf_file"]
    experiment_id = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[:4]

    # deleted indice for elastic search to make sure mappings are properly passed
    delete_indices(index="document_elasticsearch")
    delete_indices(index="label_elasticsearch")

    p = pipelines.retriever(parameters, elasticsearch_hostname, 
            elasticsearch_port,
            gpu_id = gpu_id)

    document_store = p.get_node("Retriever").document_store

    preprocessing = parameters["preprocessing"]
    split_by = parameters["split_by"]
    split_length = parameters["split_length"]
    if preprocessing:
        preprocessor = pipelines.components.preprocessors.preprocessor(split_by, split_length)
    else:
        preprocessor = None

    # Add evaluation data to Elasticsearch document store
    document_store.add_eval_data(
        evaluation_data.as_posix(),
        doc_index="document_elasticsearch",
        label_index="label_elasticsearch",
        preprocessor=preprocessor,
    )

    retriever = p.get_node("Retriever")

    if type(retriever) in [DensePassageRetriever, TitleEmbeddingRetriever, 
            EmbeddingRetriever]:
        document_store.update_embeddings(retriever,
                index="document_elasticsearch")

    if epitca_perf_file:
        expected_answers = epitca_retriever.load_perf_file_expected_answer(epitca_perf_file)
        custom_evaluation_questions = [{"query": q, "gold_ids": [a]} for q,a in
            expected_answers.items()]
        get_doc_id = lambda doc: doc.meta["id"]
    else:
        custom_evaluation_questions = None
        get_doc_id = lambda doc: doc.id

    retriever_eval_results = eval_retriever(
        document_store=document_store,
        pipeline=p,
        top_k=k,
        label_index="label_elasticsearch",
        doc_index="document_elasticsearch",
        question_label_dict_list=custom_evaluation_questions,
        get_doc_id = get_doc_id,
    )

    # Retriever Recall is the proportion of questions for which the correct document containing the answer is
    # among the correct documents
    print("Retriever Recall:", retriever_eval_results["recall"])
    # Retriever Mean Avg Precision rewards retrievers that give relevant documents a higher rank
    print("Retriever Mean Avg Precision:", retriever_eval_results["map"])

    retriever_eval_results.update(parameters)
    retriever_eval_results.update(
        {
            "date": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
            "hostname": socket.gethostname(),
            "experiment_id": experiment_id,
        }
    )

    pprint(retriever_eval_results)

    return retriever_eval_results


if __name__ == "__main__":
    load_dotenv()
    elasticsearch_hostname = os.getenv("ELASTICSEARCH_HOSTNAME") or "localhost"
    elasticsearch_port = int(os.getenv("ELASTICSEARCH_PORT")) or 9200

    result_file_path = Path("./output/results.csv")
    parameters_grid = list(ParameterGrid(param_grid=parameters))

    all_results = []
    launch_ES()
    for param in tqdm(parameters_grid, desc="GridSearch"):
        # START XP
        run_results = single_run(param, elasticsearch_hostname,
                elasticsearch_port)
        # all_results.append(run_results)
        save_results(result_file_path=result_file_path, results_list=run_results)
