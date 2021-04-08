import hashlib
import socket
from datetime import datetime
from pathlib import Path

from farm.utils import initialize_device_settings
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.pipeline import Pipeline
from haystack.retriever.dense import EmbeddingRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.preprocessor.preprocessor import PreProcessor
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
from src.evaluation.config.retriever_eval_squad_config import parameters
from src.evaluation.utils.elasticsearch_management import launch_ES, prepare_mapping, delete_indices
from src.evaluation.utils.utils_eval import eval_retriever, save_results


def single_run(parameters):
    """
        Runs a grid search config
        :param parameters: A dict with diverse config options
        :return: A dict with the results obtained running the experiment with these parameters
        """
    # col names
    evaluation_data = Path(parameters["squad_dataset"])
    retriever_type = parameters["retriever_type"]
    k = parameters["k"]
    preprocessing = parameters["preprocessing"]
    split_by=parameters["split_by"]
    split_length = parameters["split_length"]
    split_respect_sentence_boundary = parameters["split_respect_sentence_boundary"]
    experiment_id = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[:4]
    # Prepare framework

    p = Pipeline()

    prepare_mapping(SQUAD_MAPPING, preprocessing, embedding_dimension=512)

    doc_index = "document_xp"
    label_index = "label_xp"

    # deleted indice for elastic search to make sure mappings are properly passed
    delete_indices(index=doc_index)

    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by=split_by,
        split_length=split_length,
        split_overlap=0, #this must be set to 0 at the data of writting this: 22 01 2021
        split_respect_sentence_boundary=split_respect_sentence_boundary
    )

    if retriever_type == 'bm25':

        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                    create_index=False, embedding_field="emb",
                                                    embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                    custom_mapping=SQUAD_MAPPING)
        retriever = ElasticsearchRetriever(document_store=document_store)
        p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])

    elif retriever_type == "sbert":
        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                    create_index=False, embedding_field="emb",
                                                    embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                    custom_mapping=SQUAD_MAPPING)
        retriever = EmbeddingRetriever(document_store=document_store,
                                       embedding_model="distiluse-base-multilingual-cased",
                                       use_gpu=GPU_AVAILABLE, model_format="sentence_transformers",
                                       pooling_strategy="reduce_max",
                                       emb_extraction_layer=-1)
        p.add_node(component=retriever, name="SBertRetriever", inputs=["Query"])
    else:
        raise Exception(f"You chose {retriever_type}. Choose one from bm25, sbert, or dpr")



    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(evaluation_data.as_posix(), doc_index=doc_index, label_index=label_index, preprocessor=preprocessor)

    if retriever_type in ["sbert", "dpr"]:
        document_store.update_embeddings(retriever, index=doc_index)

    retriever_eval_results = eval_retriever(document_store=document_store, pipeline=p,
                                            top_k=k,
                                            label_index=label_index,
                                            doc_index=doc_index)
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
