'''
For now, it loads the config from eval_config __init__.py and uses it to start the experiments
'''

import json
import logging
import pickle
import subprocess
import time
from pathlib import Path
from random import seed
from typing import Dict, List, Tuple
from sklearn.model_selection import ParameterGrid
from argopt import argopt
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever

from src.util.convert_json_files_to_dicts import convert_json_files_to_dicts
from src.util.convert_json_to_dictsAndEmbeddings import convert_json_to_dictsAndEmbeddings

seed(42)
from tqdm import tqdm
# from haystack.database.elasticsearch import ElasticsearchDocumentStore
# from haystack.indexing.cleaning import clean_wiki_text
# from haystack.indexing.utils import convert_files_to_dicts
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever

import pandas as pd

logger = logging.getLogger(__name__)

# CONFIG
# TEST_CORPUS_PATH = "./data/25k_data/15082020-ServicePublic_QR_20170612_20180612_464_single_questions.csv"
# MODEL_TOKENIZER = "etalab-ia/camembert-base-squadFR-fquad-piaf"
# RETRIEVER = "sparse"  # sparse = bm25 ; dense = embeddings

DENSE_MAPPING = {"mappings": {"properties": {
    "link": {
        "type": "keyword"
    },
    "name": {
        "type": "keyword"
    },
    "question": {
        "type": "text"
    },
    "question_emb": {
        "type": "dense_vector",
        "dims": 512
    },
    "text": {
        "type": "text"
    }
}}}


def load_25k_test_set(test_corpus_path: str):
    """
    Loads the 25k dataset. The 25k dataset is a csv that must contain the url columns (url, url2, url3, url4)
    and a question column. The former contains the list of proposed fiches' URLs and the latter contains the
    question sent by an user.
    :param test_corpus_path: Path of the file containing the 25k corpus
    :return: Dict with the questions as key and the referenced fiches' URLs as values
    """
    url_cols = ["url", "url_2", "url_3", "url_4"]

    df = pd.read_csv(test_corpus_path).fillna("")

    dict_question_fiche = {}
    for row in df.iterrows():
        question = row[1]["question"]
        list_url = [row[1][u] for u in url_cols if row[1][u]]
        dict_question_fiche[question] = list_url
    return dict_question_fiche


def compute_retriever_precision(true_fiches, retrieved_fiches, weight_position=False):
    """
    Computes an accuracy-like score to determine the fairness of the retriever.
    Takes the k *retrieved* fiches' names and counts how many of them exist in the *true* fiches names


    :param retrieved_fiches:
    :param true_fiches:
    :return:
    """
    summed_precision = 0
    for fiche_idx, true_fiche_id in enumerate(true_fiches):
        for retrieved_doc_idx, doc in enumerate(retrieved_fiches):
            if true_fiche_id in doc:
                if weight_position:
                    summed_precision += ((fiche_idx + 1) / (fiche_idx + retrieved_doc_idx + 1))
                else:
                    summed_precision += 1
                break
    return summed_precision


# compute_retriever_precision(["f1", "f2", "f3"], ["f3"], False)

# def main(test_corpus_path: str, knowledge_base_path: str,
#          result_file_path: str, retriever_type: str,
#          k: int):
#
#
#     if len(k_range) < 2:
#         single_run(retriever, test_dataset, retriever_top_k=5, weighted_precision=False)
#     else:
#         single_run(knowledge_base_path, retriever, test_dataset, retriever_type, result_file_path, k_range,
#                    weighted_precision=True)


# def single_run(retriever, test_dataset, retriever_top_k: int, weight_position: bool = False):
#     """
#     Runs the Retriever once with the specified retriever k.
#     :param weight_position:
#     :param retriever_top_k:
#     :return:
#     """
#     mean_precision, avg_time, found_fiche, detailed_results = compute_score(retriever=retriever,
#                                                                             retriever_top_k=retriever_top_k,
#                                                                             test_dataset=test_dataset,
#                                                                             weight_position=weight_position)
#     with open(f"./results/k_{retriever_top_k}_detailed_results.json", "w") as outo:
#         json.dump(detailed_results, outo, indent=4, ensure_ascii=False)


def single_run(parameters):
    """
    Queries ES max_k - min_k times, saving at each step the results in a list. At the end plots the line
    showing the results obtained. For now we can only vary k.
    :param min_k: Minimum retriever-k to test
    :param max_k: Maximum retriever-k to test
    :param weighted_precision: Whether to take into account the position of the retrieved result in the accuracy computation
    :return:
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # col names

    test_corpus_path = Path(parameters["test_dataset"])
    knowledge_base_path = Path(parameters["knowledge_base"])
    result_file_path = Path("./results/results.csv")
    retriever_type = parameters["retriever_type"]
    k = parameters["k"]
    weighted_precision = parameters["weighted_precision"]

    # Prepare framework
    test_dataset = load_25k_test_set(test_corpus_path)
    retriever = prepare_framework(knowledge_base_path=knowledge_base_path,
                                  retriever_type=retriever_type)
    if not retriever:
        logger.info("Could not prepare the testing framework!! Exiting :(")
        return

    # All is good, let's run the experiment
    results = []
    corpus_name = knowledge_base_path.name
    tqdm.write(f"Testing k={k}")
    mean_precision, avg_time, correctly_retrieved, detailed_results_weighted = compute_score(
        retriever=retriever,
        retriever_top_k=k,
        test_dataset=test_dataset,
        weight_position=weighted_precision)
    result_dict = {"k": k,
                   "corpus": corpus_name,
                   "retriever": retriever_type,
                   "nb_documents": len(test_dataset),
                   "correctly_retrieved": correctly_retrieved,
                   "weighted_precision": str(weighted_precision),
                   "precision": mean_precision,
                   "avg_time_s": avg_time}

    results.append(result_dict)
    df_results: pd.DataFrame = pd.DataFrame(results)
    # sns.lineplot(data=df_results[["mean_precision_weighted", "mean_precision"]])
    # fig_title = f"kb={corpus_name}--k={min_k},{max_k}--retriever={retriever_type}--weighted={weight_position}"
    # plt.title(fig_title)
    #
    # plt.savefig(f"./results/{fig_title}.png")

    # if Path(result_file_path).exists():
    #     df_old = pd.read_csv(result_file_path)
    #     df_results = pd.concat([df_old, df_results])
    # with open(result_file_path, "w") as filo:
    #     df_results.to_csv(filo, index=False)
    return df_results


def prepare_framework(knowledge_base_path: str = "/data/service-public-france/extracted/",
                      retriever_type: str = "sparse"):
    """
    Loads ES if needed (check LAUNCH_ES var above) and indexes the knowledge_base corpus
    :param knowledge_base_path: PAth of the folder containing the knowledge_base corpus
    :param retriever_type: The type of retriever to be used
    :return: A Retriever object ready to be queried
    """
    try:
        # kill ES container if running
        subprocess.run(['docker stop $(docker ps -aq)'], shell=True)
        time.sleep(7)

        logging.info("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                "then set LAUNCH_ELASTICSEARCH in the script to False.")
        time.sleep(15)

        # Connect to Elasticsearch
        if retriever_type == "sparse":
            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

            dicts = convert_json_files_to_dicts(dir_path=knowledge_base_path)

            # Now, let's write the docs to our DB.
            document_store.write_documents(dicts)
            retriever = ElasticsearchRetriever(document_store=document_store)
        elif retriever_type == "dense":
            document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                        embedding_field="question_emb", embedding_dim=512,
                                                        excluded_meta_data=["question_emb"],
                                                        custom_mapping=DENSE_MAPPING)
            retriever = EmbeddingRetriever(document_store=document_store,
                                           embedding_model="distiluse-base-multilingual-cased",
                                           use_gpu=False, model_format="sentence_transformers",
                                           pooling_strategy="reduce_max")

            # dicts = convert_json_to_dictsAndEmbeddings(dir_path=knowledge_base_path,
            #                                            retriever=retriever,
            #                                            split_paragraphs=False)
            dicts = pickle.load(open("/home/pavel/code/piaf-ml/data/v11_dicts.pkl", "rb"))

            document_store.write_documents(dicts)

        return retriever
    except Exception as e:
        logger.error(f"Failed with error {str(e)}")


def compute_score(retriever: BaseRetriever, retriever_top_k: int,
                  test_dataset: Dict[str, List], weight_position: bool = False):
    """
    Given a Retriever to query and its parameters and a test dataset (couple query->true related doc), computes
    the number of matches found by the Retriever. A match is succesful if the retrieved document is among the
    true related doc of the test set.
    :param retriever: A Retriever object
    :param retriever_top_k: The number of docs to retrieve
    :param test_dataset: A collection of "query":[relevant_doc_1, relevant_doc_2, ...]
    :param weight_position: Whether to take into account the position of the retrieved result in the accuracy computation
    :return: Returns mean_precision, avg_time, and detailed_results
    """
    summed_precision = 0
    found_fiche = 0
    succeses = []
    errors = []
    if weight_position:
        logger.info("Using position weighted accuracy")
    pbar = tqdm(total=len(test_dataset))
    for question, true_fiche_urls in test_dataset.items():
        true_fiche_ids = [f.split("/")[-1] for f in true_fiche_urls]
        retrieved_results = retriever.retrieve(query=question, top_k=retriever_top_k)
        pbar.update()
        retrieved_doc_names = [f.meta["name"] for f in retrieved_results]
        precision = compute_retriever_precision(true_fiche_ids, retrieved_doc_names, weight_position=weight_position)
        summed_precision += precision
        if precision:
            found_fiche += 1
            succeses.append({"question": question,
                             "pred_fiche": retrieved_doc_names,
                             "true_fiche": true_fiche_ids,
                             })
        else:
            errors.append({"question": question,
                           "pred_fiche": retrieved_doc_names,
                           "true_fiche": true_fiche_ids,
                           })

    avg_time = pbar.avg_time
    pbar.close()
    detailed_results = {"successes": succeses, "errors": errors, "avg_time": avg_time}

    mean_precision = summed_precision / len(test_dataset)
    tqdm.write(
        f"The retriever correctly found {found_fiche} fiches among {len(test_dataset)}. Mean_precision {mean_precision}. "
        f"Time per ES query (ms): {avg_time * 1000:.3f}")
    return mean_precision, avg_time, found_fiche, detailed_results


if __name__ == '__main__':
    from src.evaluation.eval_config import parmeters

    parameters_grid = list(ParameterGrid(param_grid=parmeters))
    grid_dfs = []
    for parameters in tqdm(parameters_grid, desc="GridSearch"):
        # START XP
        df_results = single_run(parameters)
        grid_dfs.append(df_results)


