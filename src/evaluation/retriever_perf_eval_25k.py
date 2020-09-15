import json
import logging
import subprocess
import time
from random import sample, seed
seed(42)
from tqdm import tqdm
from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever

import pandas as pd
logger = logging.getLogger(__name__)

LAUNCH_ELASTICSEARCH = False

# Load 25k dataset
PATH_25k = "./data/25k_data/15082020-ServicePublic_QR_20170612_20180612_464_single_questions.csv"
def load_25k_dataset():
    url_cols = ["url", "url_2", "url_3", "url_4"]

    df = pd.read_csv(PATH_25k).fillna("")

    dict_question_fiche = {}
    for row in df.iterrows():
        question = row[1]["question"]
        list_url = [row[1][u] for u in url_cols if row[1][u]]
        dict_question_fiche[question] = list_url
    return dict_question_fiche


def compute_retriever_precision(true_fiches, retrieved_fiches, at_k=False):
    """
    Computes an accuracy-like score to determine the fairness of the retriever.
    Takes the k *retrieved* fiches' names and counts how many of them exist in the *true* fiches names


    :param retrieved_fiches:
    :param true_fiches:
    :return:
    """
    summed_precision = 0
    for fiche_idx, true_fiche_id in enumerate(true_fiches):
        for doc_idx, doc in enumerate(retrieved_fiches):
            if true_fiche_id in doc:
                if at_k:
                    summed_precision += (fiche_idx + 1) / (doc_idx + 1)
                else:
                    summed_precision += 1
                break
    if at_k:
        summed_precision = summed_precision / len(true_fiches)
    return summed_precision


if LAUNCH_ELASTICSEARCH:
    logging.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
    )
    if status.returncode:
        raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                        "then set LAUNCH_ELASTICSEARCH in the script to False.")
    time.sleep(15)

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

doc_dir = "/data/service-public-france/extracted/"

dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the docs to our DB.
if LAUNCH_ELASTICSEARCH:
    document_store.write_documents(dicts)
else:
    logger.warning("Since we already have a running ES instance we should not index the same documents again. \n"
                   "If you still want to do this call: document_store.write_documents(dicts) manually ")


retriever = ElasticsearchRetriever(document_store=document_store)
reader = TransformersReader(model="etalab-ia/camembert-base-squadFR-fquad-piaf",
                            tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf", use_gpu=-1)
# reader = TransformersReader(model="fmikaelian/camembert-base-fquad",
#                             tokenizer="fmikaelian/camembert-base-fquad", use_gpu=-1)
finder = Finder(reader, retriever)

test_qr_dataset = load_25k_dataset()

summed_precision = 0
found_fiche = 0
results = []
TOP_K = 10
for question, true_fiche_urls in tqdm(test_qr_dataset.items()):
    true_fiche_ids = [f.split("/")[-1] for f in true_fiche_urls]
    retrieved_results = retriever.retrieve(query=question, top_k=TOP_K)
    retrieved_doc_names = [f.meta["name"] for f in retrieved_results]
    precision = compute_retriever_precision(true_fiche_ids, retrieved_doc_names, at_k=False)
    summed_precision += precision
    if precision:
        found_fiche += 1
    # first_answer_score = retriever_results[0].query_score
    # true if we found any of the true fiches within the retriever answers



    # logger.info(f"SUCCESS: Q:{question},  F:{true_fiche_ids},   P:{first_answer_name}")
    # results.append({"question": question,
    #                 "pred_fiche": first_answer_name,
    #                 "true_fiche": "|".join(true_fiche_ids),
    #                 "score": first_answer_score
    #                 })

    pass
mean_precision = summed_precision / len(test_qr_dataset)

tqdm.write(f"The retriever correctly found {found_fiche} fiches among {len(test_qr_dataset)}. Accuracy:{mean_precision}")

# print_answers(prediction, details="all")
# json.dump(results, open("./data/evaluation_haystack.json", "w"), indent=4, ensure_ascii=False)