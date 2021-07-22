import hashlib
import socket
from datetime import datetime
from pathlib import Path

from farm.utils import initialize_device_settings
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
from src.evaluation.config.title_qa_pipeline_config import parameters
from src.evaluation.utils.pipelines.custom_pipelines import TitleQAPipeline
from src.evaluation.utils.elasticsearch_management import (delete_indices,
                                                           launch_ES,
                                                           prepare_mapping)
from src.evaluation.utils.TitleEmbeddingRetriever import \
    TitleEmbeddingRetriever
from src.evaluation.utils.utils_eval import eval_titleQA_pipeline, save_results


def single_run(parameters):
    """
    Runs a grid search config 

    :param parameters: A dict with diverse config options 
    :return: A dict with the results obtained running the experiment with these parameters
    """
    # col names
    evaluation_data = Path(parameters["squad_dataset"])
    k_retriever = parameters["k_retriever"]
    preprocessing = parameters["preprocessing"]
    experiment_id = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[:4]

    # Prepare framework
    prepare_mapping(
        mapping=SQUAD_MAPPING, title_boosting_factor=1, embedding_dimension=768
    )

    doc_index = "document_faq"
    label_index = "label_faq"

    document_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index=doc_index,
        create_index=False,
        embedding_field="emb",
        embedding_dim=768,
        excluded_meta_data=["emb"],
        similarity="cosine",
        custom_mapping=SQUAD_MAPPING,
    )

    retriever = TitleEmbeddingRetriever(
        document_store=document_store,
        embedding_model="distilbert-base-multilingual-cased",
        model_version="1a01b38498875d45f69b2a6721bf6fe87425da39",
        use_gpu=GPU_AVAILABLE,
        model_format="transformers",
        pooling_strategy="reduce_max",
        emb_extraction_layer=-1,
    )

    p = TitleQAPipeline(retriever)

    # Add evaluation data to Elasticsearch document store
    # We first delete the custom tutorial indices to not have duplicate elements
    # make sure these indices do not collide with existing ones, the indices will be wiped clean before data is inserted

    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(
        evaluation_data.as_posix(), doc_index=doc_index, label_index=label_index
    )

    document_store.update_embeddings(retriever, index=doc_index)

    retriever_eval_results = eval_titleQA_pipeline(
        document_store=document_store,
        pipeline=p,
        k_retriever=k_retriever,
        label_index=label_index,
    )

    print("Reader Accuracy:", retriever_eval_results["reader_topk_accuracy"])
    print("reader_topk_f1:", retriever_eval_results["reader_topk_f1"])

    retriever_eval_results.update(parameters)
    retriever_eval_results.update(
        {
            "date": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
            "hostname": socket.gethostname(),
            "experiment_id": experiment_id,
        }
    )

    # deleted indice for elastic search to make sure mappings are properly passed
    delete_indices(index=doc_index)

    return retriever_eval_results


if __name__ == "__main__":
    result_file_path = Path("./output/results.csv")
    parameters_grid = list(ParameterGrid(param_grid=parameters))

    device, n_gpu = initialize_device_settings(use_cuda=True)
    GPU_AVAILABLE = 1 if device == "gpu" else 0

    all_results = []
    launch_ES()
    for param in tqdm(parameters_grid, desc="GridSearch"):
        # START XP
        run_results = single_run(param)
        # all_results.append(run_results)
        save_results(result_file_path=result_file_path, results_list=run_results)
