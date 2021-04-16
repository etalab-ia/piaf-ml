import torch
import time

from pathlib import Path
from tqdm import tqdm

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever
from haystack.reader.transformers import TransformersReader
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.pipeline import Pipeline, ExtractiveQAPipeline

from farm.utils import initialize_device_settings
from sklearn.model_selection import ParameterGrid

from src.evaluation.config.retriever_reader_eval_squad_config import parameters
from src.evaluation.utils.TitleEmbeddingRetriever import TitleEmbeddingRetriever
from src.evaluation.utils.elasticsearch_management import launch_ES, delete_indices, prepare_mapping
from src.evaluation.utils.mlflow_management import prepare_mlflow_server, add_extra_params, create_run_ids
from src.evaluation.utils.utils_eval import eval_retriever_reader, save_results
from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
from src.evaluation.utils.custom_pipelines import TitleBM25QAPipeline
import mlflow
from mlflow.tracking import MlflowClient

prepare_mlflow_server()

GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    n_gpu = torch.cuda.current_device()
else:
    n_gpu = -1


def single_run(parameters):
    """
        Queries ES max_k - min_k times, saving at each step the results in a list. At the end plots the line
        showing the results obtained. For now we can only vary k.
        :param min_k: Minimum retriever-k to test
        :param max_k: Maximum retriever-k to test
        :param weighted_precision: Whether to take into account the position of the retrieved result in the accuracy computation
        :return:
        """
    # col names
    evaluation_data = Path(parameters["squad_dataset"])
    retriever_type = parameters["retriever_type"]
    k_retriever = parameters["k_retriever"]
    k_title_retriever = parameters["k_title_retriever"]
    k_reader_per_candidate = parameters["k_reader_per_candidate"]
    k_reader_total = parameters["k_reader_total"]
    preprocessing = parameters["preprocessing"]
    split_by = parameters["split_by"]
    split_length = parameters["split_length"]
    title_boosting_factor = parameters["boosting"]

    doc_index = "document_xp"
    label_index = "label_xp"

    # deleted indice for elastic search to make sure mappings are properly passed
    delete_indices(index=doc_index)

    prepare_mapping(mapping=SQUAD_MAPPING, preprocessing=preprocessing, title_boosting_factor=title_boosting_factor,
                    embedding_dimension=512)

    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by=split_by,
        split_length=split_length,
        split_overlap=0,  # this must be set to 0 at the data of writting this: 22 01 2021
        split_respect_sentence_boundary=False  # the support for this will soon be removed : 29 01 2021
    )

    reader = TransformersReader(model_name_or_path="etalab-ia/camembert-base-squadFR-fquad-piaf",
                                tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf",
                                use_gpu=gpu_id, top_k_per_candidate=k_reader_per_candidate)

    if retriever_type == 'bm25':

        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                    search_fields=["name", "text"],
                                                    create_index=False, embedding_field="emb",
                                                    scheme="",
                                                    embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                    custom_mapping=SQUAD_MAPPING)
        retriever = ElasticsearchRetriever(document_store=document_store)
        p = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    elif retriever_type == "sbert":
        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                    search_fields=["name", "text"],
                                                    create_index=False, embedding_field="emb",
                                                    embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                    custom_mapping=SQUAD_MAPPING)
        retriever = EmbeddingRetriever(document_store=document_store,
                                       embedding_model="distiluse-base-multilingual-cased",
                                       use_gpu=GPU_AVAILABLE, model_format="sentence_transformers",
                                       pooling_strategy="reduce_max",
                                       emb_extraction_layer=-1)
        p = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    elif retriever_type == "title_bm25":
        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index,
                                                    search_fields=["name", "text"],
                                                    create_index=False, embedding_field="emb",
                                                    embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                    custom_mapping=SQUAD_MAPPING)
        retriever = TitleEmbeddingRetriever(document_store=document_store,
                                            embedding_model="distiluse-base-multilingual-cased",
                                            use_gpu=GPU_AVAILABLE, model_format="sentence_transformers",
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
                                            embedding_model="distiluse-base-multilingual-cased",
                                            use_gpu=GPU_AVAILABLE, model_format="sentence_transformers",
                                            pooling_strategy="reduce_max",
                                            emb_extraction_layer=-1)
        p = ExtractiveQAPipeline(reader=reader, retriever=retriever)


    else:
        raise Exception(f"You chose {retriever_type}. Choose one from bm25, sbert, dpr or title_bm25")

    # Add evaluation data to Elasticsearch document store
    # We first delete the custom tutorial indices to not have duplicate elements
    # make sure these indices do not collide with existing ones, the indices will be wiped clean before data is inserted

    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(evaluation_data.as_posix(), doc_index=doc_index, label_index=label_index,
                                 preprocessor=preprocessor)

    if retriever_type in ["sbert", "dpr", "title_bm25", "title"]:
        document_store.update_embeddings(retriever, index=doc_index)

    start = time.time()
    retriever_eval_results = eval_retriever_reader(document_store=document_store, pipeline=p,
                                                   k_retriever=k_retriever, k_reader_total=k_reader_total,
                                                   label_index=label_index)
    end = time.time()
    time_per_label = (end - start) / document_store.get_label_count(index=label_index)

    print("Reader Accuracy:", retriever_eval_results["reader_topk_accuracy"])
    print("reader_topk_f1:", retriever_eval_results["reader_topk_f1"])

    retriever_eval_results.update({"time_per_label": time_per_label})

    return retriever_eval_results


if __name__ == '__main__':
    result_file_path = Path("./results/results_reader.csv")
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
    list_run_ids = create_run_ids(parameters_grid)
    mlflow.set_experiment(experiment_name=experiment_name)

    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    list_past_run_names = [client.get_run(run.run_id).data.tags['mlflow.runName']for run in client.list_run_infos(experiment_id)]

    for idx, param in zip(list_run_ids, tqdm(parameters_grid, desc="GridSearch", unit="config")):
        add_extra_params(param)

        if idx not in list_past_run_names:
            tqdm.write(f"Doing run with config : {param}")
            try:
                with mlflow.start_run(run_name=idx) as run:
                    mlflow.log_params(param)
                    # START XP
                    run_results = single_run(param)
                    mlflow.log_metrics({k: v for k, v in run_results.items() if v is not None})
                run_results.update(param)
                save_results(result_file_path=result_file_path, results_list=run_results)
            except Exception as e:
                Exception(f"Could not run this config: {param}")
                tqdm.write(f"Error:{e}")
                continue
        else:
            print('config already done')