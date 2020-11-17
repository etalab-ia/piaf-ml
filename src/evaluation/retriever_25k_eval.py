'''
For now, it loads the config from eval_config __init__.py and uses it to start the experiments
'''
import json
import logging
import os
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path
from random import seed
from typing import Dict, List, Tuple
import hashlib

import torch
from elasticsearch import Elasticsearch
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from sklearn.model_selection import ParameterGrid
from src.evaluation.eval_config import parameters
from src.util.convert_json_files_to_dicts import convert_json_files_to_dicts, convert_json_files_v10_to_dicts
from src.util.convert_json_to_dictsAndEmbeddings import convert_json_to_dicts

seed(42)
from tqdm import tqdm
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever
from haystack.document_store.faiss import FAISSDocumentStore

import pandas as pd
import socket

logger = logging.getLogger(__name__)

GPU_AVAILABLE = torch.cuda.is_available()
USE_CACHE = True
SBERT_MAPPING = {"mappings": {"properties": {
    "link": {
        "type": "keyword"
    },
    "name": {
        "type": "keyword"
    },
    "question_sparse": {
        "type": "text"
    },
    "question_emb": {
        "type": "dense_vector",
        "dims": 512
    },
    "text": {
        "type": "text"
    },
    "theme": {
        "type": "keyword"
    },
    "dossier": {
        "type": "keyword"
    }
}}}

DPR_MAPPING = {"mappings": {"properties": {
    "link": {
        "type": "keyword"
    },
    "name": {
        "type": "keyword"
    },
    "question_sparse": {
        "type": "text"
    },
    "question_emb": {
        "type": "dense_vector",
        "dims": 768
    },
    "text": {
        "type": "text"
    },
    "theme": {
        "type": "keyword"
    },
    "dossier": {
        "type": "keyword"
    }
}}}

SPARSE_MAPPING = {"mappings": {"properties": {
    "question_sparse": {
        "type": "text"
    },
    "text": {
        "type": "text"
    },
    "theme": {
        "type": "keyword"
    },
    "dossier": {
        "type": "keyword"
    }
}}}


def load_25k_test_set(test_corpus_path: str):
    """
    Loads the 25k dataset. The 25k dataset is a csv that must contain the url columns (url, url2, url3, url4)
    and a question column. The former contains the list of proposed fiches' URLs and the latter contains the
    question sent by an user.
    :param corpus_path: Path of the file containing the 25k corpus
    :return: Dict with the questions as key and a meta dict as values,
    the meta dict containing urls where the answer is and the arborescence of where the answer lies
    """
    url_cols = ["url", "url_2", "url_3", "url_4"]

    df = pd.read_csv(test_corpus_path).fillna("")
    dict_question_fiche = {}
    for row in df.iterrows():
        question = row[1]["question"]
        list_url = [row[1][u] for u in url_cols if row[1][u]]
        arbo = {'theme': row[1]['theme'],
                'dossier': row[1]['dossier']}
        meta = {'urls': list_url, 'arbo': arbo}
        dict_question_fiche[question] = meta

    return dict_question_fiche


def compute_retriever_precision(true_fiches, retrieved_results, weight_position=False):
    """
    Computes an accuracy-like score to determine the fairness of the retriever.
    Takes the k *retrieved* fiches' names and counts how many of them exist in the *true* fiches names


    :param retrieved_fiches:
    :param true_fiches:
    :param weight_position: Bool indicates if the precision must be calculated with a weighted precision
    :return:
    """
    retrieved_docs = []
    summed_precision = 0
    results_info = {}
    correct_doc_info = {}
    retrieved_doc_names = [(f.meta["name"],
                            idx + 1,
                            f.score,
                            f.probability) for idx, f in enumerate(retrieved_results)]
    for fiche_idx, true_fiche_id in enumerate(true_fiches):
        for retrieved_doc_idx, retrieved_doc in enumerate(retrieved_results):
            retrieved_doc_id = retrieved_doc.meta["name"]
            retrieved_docs.append(retrieved_doc_id)
            if true_fiche_id in retrieved_doc_id:
                if weight_position:
                    summed_precision += ((fiche_idx + 1) / (fiche_idx + retrieved_doc_idx + 1))
                else:
                    summed_precision += 1
                correct_doc_info[retrieved_doc_id] = {
                    "position": retrieved_doc_idx + 1,
                    "probability": retrieved_doc.probability,
                    "score": retrieved_doc.score}
                break

    results_info["true_fiches"] = true_fiches
    results_info["pred_fiches"] = retrieved_doc_names

    return summed_precision, results_info


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
    test_corpus_path = Path(parameters["test_dataset"])
    knowledge_base_path = Path(parameters["knowledge_base"])
    retriever_type = parameters["retriever_type"]
    k = parameters["k"]
    weighted_precision = parameters["weighted_precision"]
    filter_level = parameters["filter_level"]

    experiment_id = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[:4]
    # Prepare framework
    test_dataset = load_25k_test_set(test_corpus_path)
    retriever = load_retriever(knowledge_base_path=knowledge_base_path,
                               retriever_type=retriever_type)

    if not retriever:
        raise Exception("Could not prepare the testing framework!! Exiting :(")

    # All is good, let's run the experiment
    results = []
    tqdm.write(str(parameters))
    mean_precision, avg_time, correctly_retrieved, detailed_results_weighted, nb_questions = compute_score(
        retriever=retriever,
        retriever_top_k=k,
        test_dataset=test_dataset,
        weight_position=weighted_precision,
        filter_level=filter_level
    )

    results_dict = dict(parameters)
    results_dict.update({
        "nb_documents": nb_questions,
        "correctly_retrieved": correctly_retrieved,
        "precision": mean_precision,
        "avg_time_s": avg_time,
        "date": datetime.today().strftime('%Y-%m-%d_%H-%M-%S'),
        "hostname": socket.gethostname(),
        "experiment_id": experiment_id})

    results.append(results_dict)
    df_results: pd.DataFrame = pd.DataFrame(results)
    detailed_results_weighted["experiment_id"] = experiment_id

    ordered_headers = ["experiment_id",
                       "knowledge_base", "test_dataset", "k", "filter_level", "retriever_type",
                       "nb_documents", "correctly_retrieved", "weighted_precision",
                       "precision", "avg_time_s", "date", "hostname"]

    df_results = df_results[ordered_headers]

    return df_results, detailed_results_weighted


def save_results(result_file_path: Path, all_results: List[Tuple]):
    grid_dfs, detailed_results = zip(*all_results)
    df_results = pd.concat(grid_dfs)
    if result_file_path.exists():
        df_old = pd.read_csv(result_file_path)
        df_results = pd.concat([df_old, df_results])
    else:
        os.makedirs(result_file_path.parent)
    with open(result_file_path.as_posix(), "w") as filo:
        df_results.to_csv(filo, index=False)
    # saved detailed results
    for dic in detailed_results:
        file_name = f"{dic['experiment_id']}_detailed_results.json"
        with open((result_file_path.parent / file_name).as_posix(), "w") as filo:
            json.dump(dic, filo, indent=4, ensure_ascii=False)


def launch_ES():
    es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)
    if not es.ping():
        logging.info("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                "then set LAUNCH_ELASTICSEARCH in the script to False.")


def load_cached_dict_embeddings(knowledge_base_path: Path, retriever_type: str,
                                cached_dicts_path: Path = Path("./data/dense_dicts/")):
    cached_dicts_name = cached_dicts_path / Path(f"{knowledge_base_path.name}_{retriever_type}.pkl")
    if cached_dicts_name.exists():
        logger.info(f"Found and loading embeddings dict cache: {cached_dicts_name}")
        try:
            with open(cached_dicts_name, "rb") as cache:
                dict_embeddings = pickle.load(cache)
            return dict_embeddings
        except Exception as e:
            logger.info(f"Could not load dict embeddings {cached_dicts_name}. Error: {str(e)}")
            return
    else:
        return


def cache_dict_embeddings(dicts: Dict, knowledge_base_path: Path, retriever_type: str,
                          cached_dicts_path: Path = Path("./data/dense_dicts/")):
    cached_dicts_name = cached_dicts_path / Path(f"{knowledge_base_path.name}_{retriever_type}.pkl")

    with open(cached_dicts_name, "wb") as cache:
        pickle.dump(dicts, cache)


def load_retriever(knowledge_base_path: str = "/data/service-public-france/extracted/",
                   retriever_type: str = "sparse"):
    """
    Loads ES if needed (check LAUNCH_ES var above) and indexes the knowledge_base corpus
    :param knowledge_base_path: PAth of the folder containing the knowledge_base corpus
    :param retriever_type: The type of retriever to be used
    :return: A Retriever object ready to be queried
    """
    retriever = None
    try:
        # delete the index to make sure we are not using other docs
        es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)
        es.indices.delete(index='document', ignore=[400, 404])
        if retriever_type == "bm25":

            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                        search_fields=['question_sparse'],
                                                        text_field='text',
                                                        custom_mapping=SPARSE_MAPPING)

            retriever = ElasticsearchRetriever(document_store=document_store)

            dicts = convert_json_to_dicts(dir_path=knowledge_base_path,
                                          retriever=retriever,
                                          compute_embeddings=False)

            # Now, let's write the docs to our DB.
            document_store.write_documents(dicts)

        elif retriever_type == "sbert":

            # TODO: change the way embedding_dim is declared as it may vary based on the embedding_model

            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                        search_fields=['question_sparse'],
                                                        embedding_field="question_emb", embedding_dim=512,
                                                        excluded_meta_data=["question_emb"],
                                                        custom_mapping=SBERT_MAPPING)

            retriever = EmbeddingRetriever(document_store=document_store,
                                           embedding_model="/data/models/dpr/distiluse",
                                           use_gpu=GPU_AVAILABLE, model_format="sentence_transformers",
                                           pooling_strategy="reduce_max")
            dicts = []
            if USE_CACHE:
                dicts = load_cached_dict_embeddings(knowledge_base_path=Path(knowledge_base_path),
                                                    retriever_type=retriever_type)
            if not dicts:
                dicts = convert_json_to_dicts(dir_path=knowledge_base_path,
                                              retriever=retriever,
                                              compute_embeddings=True)

                cache_dict_embeddings(dicts=dicts, knowledge_base_path=Path(knowledge_base_path),
                                      retriever_type=retriever_type)

            document_store.write_documents(dicts)

        elif retriever_type == "dpr":

            # status = subprocess.run(
            #     ['docker run --name haystack-postgres -p 5432:5432 -e POSTGRES_PASSWORD=password -d postgres'],
            #     shell=True)
            # time.sleep(3)
            # status = subprocess.run(
            # ['docker exec -it haystack-postgres psql -U postgres -c "CREATE DATABASE haystack;"'], shell=True)
            # time.sleep(1)
            # document_store = FAISSDocumentStore(sql_url="postgresql://postgres:password@localhost:5432/haystack")
            # document_store = FAISSDocumentStore()
            #
            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                        search_fields=['question_sparse'],
                                                        embedding_field="question_emb", embedding_dim=768,
                                                        excluded_meta_data=["question_emb"],
                                                        custom_mapping=DPR_MAPPING)

            retriever = DensePassageRetriever(document_store=document_store,
                                              query_embedding_model="/home/pavel/code/dpr2hf/models/encoder_question",
                                              passage_embedding_model="/home/pavel/code/dpr2hf/models/encoder_ctx",
                                              use_gpu=GPU_AVAILABLE,
                                              embed_title=False,
                                              max_seq_len_passage=500,
                                              batch_size=16,
                                              use_fast_tokenizers=False
                                              )
            # TODO: Embed passages check function here
            dicts = []
            if USE_CACHE:
                dicts = load_cached_dict_embeddings(knowledge_base_path=Path(knowledge_base_path),
                                                    retriever_type=retriever_type)

            if not dicts:
                dicts = convert_json_to_dicts(dir_path=knowledge_base_path,
                                              retriever=retriever,
                                              compute_embeddings=True)

                cache_dict_embeddings(dicts=dicts, knowledge_base_path=Path(knowledge_base_path),
                                      retriever_type=retriever_type)

            # document_store.faiss_index.reset()
            document_store.write_documents(dicts)
        else:
            raise Exception("Choose a retriever type between : bm25, sbert, dpr")
        return retriever
    except Exception as e:
        logger.error(f"Failed with error {str(e)}")
        return retriever


def compute_score(retriever: BaseRetriever, retriever_top_k: int,
                  test_dataset: Dict[str, List], weight_position: bool = False, filter_level: str = None):
    """
    Given a Retriever to query and its parameters and a test dataset (couple query->true related doc), computes
    the number of matches found by the Retriever. A match is succesful if the retrieved document is among the
    true related doc of the test set.
    :param retriever: A Retriever object
    :param retriever_top_k: The number of docs to retrieve
    :param test_dataset: A collection of "query":[relevant_doc_1, relevant_doc_2, ...]
    :param weight_position: Whether to take into account the position of the retrieved result in the accuracy computation
    :param filter_level: The name of the filter requested, usually the level from the arborescence : 'theme', 'dossier' ..
    :return: Returns mean_precision, avg_time, and detailed_results
    """
    summed_precision = 0
    found_fiche = 0
    nb_questions = 0
    successes = {}
    errors = {}
    if weight_position:
        logger.info("Using position weighted accuracy")
    pbar = tqdm(total=len(test_dataset))
    for question, meta in test_dataset.items():
        true_fiche_urls = meta['urls']
        true_fiche_ids = [f.split("/")[-1] for f in true_fiche_urls]
        if filter_level is None:
            retrieved_results = retriever.retrieve(query=question, top_k=retriever_top_k)
        else:
            arborescence = meta['arbo']
            filter_value = arborescence[filter_level]
            if filter_level is not None and filter_value == '':  # sometimes the value for the filter is not present in the data
                continue
            retrieved_results = retriever.retrieve(query=question, filters={filter_level: [filter_value]},
                                                   top_k=retriever_top_k)
        pbar.update()

        precision, results_info = compute_retriever_precision(true_fiche_ids,
                                                              retrieved_results,
                                                              weight_position=weight_position)

        summed_precision += precision
        nb_questions += 1
        if precision:
            found_fiche += 1
            successes[question] = results_info
        else:
            errors[question] = results_info
    avg_time = pbar.avg_time
    if avg_time is None:  # quick fix for a bug idk why is happening
        avg_time = 0
    pbar.close()
    detailed_results = {"successes": successes, "errors": errors, "avg_time": avg_time}

    mean_precision = summed_precision / nb_questions
    tqdm.write(
        f"The retriever correctly found {found_fiche} fiches among {nb_questions}. Mean_precision {mean_precision}. "
        f"Time per ES query (ms): {avg_time * 1000:.3f}")
    return mean_precision, avg_time, found_fiche, detailed_results, nb_questions


if __name__ == '__main__':

    result_file_path = Path("./results/results.csv")
    parameters_grid = list(ParameterGrid(param_grid=parameters))
    all_results = []
    launch_ES()
    for param in tqdm(parameters_grid, desc="GridSearch"):
        # START XP
        run_results = single_run(param)
        all_results.append(run_results)

    save_results(result_file_path=result_file_path, all_results=all_results)
