import hashlib
import socket
from datetime import datetime
from pathlib import Path

from farm.utils import initialize_device_settings
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.preprocessor.preprocessor import PreProcessor
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
from src.evaluation.config.FAQstyle_config import parameters
from src.evaluation.utils.elasticsearch_management import launch_ES, prepare_mapping
from src.evaluation.utils.utils_eval import eval_faq_pipeline, save_results
from src.evaluation.utils.FAQEmbeddingRetriever import FAQEmbeddingRetriever
from src.evaluation.utils.FAQPipeline import FAQPipeline



def single_run(parameters):
    """
    Runs a grid search config
    :param parameters: A dict with diverse config options
    :return: A dict with the results obtained running the experiment with these parameters
    """
    # col names
    evaluation_data = Path(parameters["squad_dataset"])
    retriever_type = parameters["retriever_type"]
    k_retriever = parameters["k_retriever"]
    preprocessing = parameters["preprocessing"]
    split_by=parameters["split_by"]
    split_length = parameters["split_length"]
    split_respect_sentence_boundary = parameters["split_respect_sentence_boundary"]
    experiment_id = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[:4]
    # Prepare framework


    prepare_mapping(SQUAD_MAPPING, preprocessing, embedding_dimension=512)

    doc_index = "document_faq"
    label_index = "label_faq"

    if retriever_type == 'bm25':

        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                    create_index=False, embedding_field="emb",
                                                    embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                    custom_mapping=SQUAD_MAPPING)
        retriever = ElasticsearchRetriever(document_store=document_store)
    elif retriever_type == "sbert":
        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                    create_index=False, embedding_field="emb",
                                                    embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                    custom_mapping=SQUAD_MAPPING)
        retriever = FAQEmbeddingRetriever(document_store=document_store,
                                       embedding_model="distiluse-base-multilingual-cased",
                                       use_gpu=GPU_AVAILABLE, model_format="sentence_transformers",
                                       pooling_strategy="reduce_max",
                                       emb_extraction_layer=-1)
    else:
        raise Exception(f"You chose {retriever_type}. Choose one from bm25, sbert, or dpr")

    p = FAQPipeline(retriever)

    # Add evaluation data to Elasticsearch document store
    # We first delete the custom tutorial indices to not have duplicate elements
    # make sure these indices do not collide with existing ones, the indices will be wiped clean before data is inserted

    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(evaluation_data.as_posix(), doc_index=doc_index, label_index=label_index)

    if retriever_type in ["sbert", "dpr"]:
        document_store.update_embeddings(retriever, index=doc_index)

    retriever_eval_results = eval_faq_pipeline(document_store=document_store, pipeline=p,
                                                   k_retriever=k_retriever,
                                                   label_index=label_index)
    # Retriever Recall is the proportion of questions for which the correct document containing the answer is
    # among the correct documents
    print("Retriever Recall:", retriever_eval_results["recall"])
    # Retriever Mean Avg Precision rewards retrievers that give relevant documents a higher rank
    print("Retriever Mean Avg Precision:", retriever_eval_results["map"])

    retriever_eval_results.update(parameters)
    retriever_eval_results.update({"date": datetime.today().strftime('%Y-%m-%d_%H-%M-%S'),
                                   "hostname": socket.gethostname(),
                                   "experiment_id": experiment_id})
    return retriever_eval_results


if __name__ == '__main__':
    result_file_path = Path("./results/results.csv")
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