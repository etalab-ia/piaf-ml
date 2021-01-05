"""
Script to convert a SQuAD-like QA-dataset format JSON file to DPR Dense Retriever training format

Usage:
    squad2dpr.py <squad_file_path> <dpr_output_path> [options]

Arguments:
    <squad_file_path>   SQuAD file path
    <dpr_output_path>   DPR outpput folder path
"""
import logging
import subprocess
from pathlib import Path
from time import sleep
from typing import List, Dict, Iterator

from argopt import argopt
from elasticsearch import Elasticsearch
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from tqdm import tqdm
import json
import random
import re
import base64
random.seed(42)
import binascii
"""
[
    {
        "question": "....",
        "answers": ["...", "...", "..."],
        "positive_ctxs": [{
            "title": "...",
            "text": "...."
        }],
        "negative_ctxs": ["..."],
        "hard_negative_ctxs": ["..."]
    },
    ...
]
"""


def convert_squad_to_dicts(squad_data: dict, max_char_size=600):
    # documents.append({"text": text, "meta": {"name": path.name}})
    documents = []
    for article in squad_data[:]:
        article_title = article["title"]
        for para_idx, paragraph in enumerate(article["paragraphs"]):
            context = paragraph["context"]
            if len(context) > max_char_size:
                end_pos = get_near_entire_phrase(context, max_char_size, side="right")
                context = context[:end_pos]
            documents.append({"text": context, "meta": {"name": f"{article_title}_{para_idx}"}})
    return documents


def launch_and_index_es(documents_dicts: List):
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
        sleep(7)

    es.indices.delete(index='document', ignore=[400, 404])
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store.write_documents(documents_dicts)
    retriever = ElasticsearchRetriever(document_store=document_store)
    return retriever


def prepare_es_retrieval(squad_data: Dict):
    documents = convert_squad_to_dicts(squad_data=squad_data)
    return launch_and_index_es(documents)


def limit_context_size(text_list, context: str, max_char_size=600):
    """
    Takes a context string and limits its size to around 100 words per context.
    As I do not want to deal for now with spans and tokenization and machin, I will assume
    a word is 5 chars in average. So a context will have max 500 size by default.
    It considers the answer (of length q), and grabs the n previous characters and the m next characters so that
    q+m+n =~ max_char_size

    :param max_char_size: Max size of the tet chink (in chars)
    :param text_list: list containing the answers' infos [{"answer_start" : "...", "text" : "..."}]
    :param context: The wikipedia paragraph where the answer is found
    :return: A list of limited size contexts
    """
    seen_answers = []
    limited_contexts = []
    for answer in text_list:
        answer_text = answer["text"]

        if answer_text in seen_answers:
            continue
        context_len = len(context)

        if context_len <= max_char_size:
            limited_contexts.append(context)

        answer_start_pos = int(answer["answer_start"])
        answer_len = len(answer_text)
        answer_end_pos = answer_start_pos + answer_len
        required_chunk_len = (max_char_size - answer_len) // 2
        left_chunk_end_pos = answer_start_pos - 1
        if left_chunk_end_pos > required_chunk_len:
            left_chunk_start_pos = left_chunk_end_pos - required_chunk_len
            len_right_chunk = required_chunk_len
        else:
            left_chunk_start_pos = 0
            remaining = required_chunk_len - left_chunk_end_pos
            len_right_chunk = required_chunk_len + remaining
        right_chunk_start_pos = answer_end_pos + 1
        right_chunk_end_pos = min(len(context) - 1, right_chunk_start_pos + len_right_chunk)

        left_chunk_start_pos = get_near_entire_phrase(context, left_chunk_start_pos, side="left")
        right_chunk_end_pos = get_near_entire_phrase(context, right_chunk_end_pos, side="right")
        seen_answers.append(answer_text)
        limited_context = context[left_chunk_start_pos:right_chunk_end_pos]
        limited_contexts.append(limited_context)
    return list(set(limited_contexts))


def get_near_entire_phrase(context: str, pos: int, side="left"):
    re_upper = re.compile(r"\b[A-Z]")
    re_period = re.compile(r"\w\s?\.(\s|$)")
    if side == "left":
        if context[pos].isupper():
            return pos
        find_uppercase = list(re_upper.finditer(context[:pos]))
        if find_uppercase:
            nearest_previous_uppercase = find_uppercase[-1].start()
            return nearest_previous_uppercase
        else:
            return pos
    elif side == "right":
        if context[pos] == ".":  # TODO: If we have "M. Something" this will break
            return pos
        find_period = list(re_period.finditer(context[pos:]))
        if find_period:
            nearest_next_period = find_period[0].end()
            return pos + nearest_next_period
        else:
            return pos


def create_dpr_training_dataset(squad_file_path: Path):
    squad_file = json.load(open(squad_file_path.as_posix()))
    version = squad_file["version"]
    squad_data = squad_file["data"]
    retriever = prepare_es_retrieval(squad_data=squad_data)
    random.shuffle(squad_data)
    list_DPR = []
    n_questions = 0
    n_non_added_questions = 0
    for idx_article, article in enumerate(tqdm(squad_data[:], unit="article")):
        article_title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                answers = [a["text"] for a in question["answers"]]
                hard_negative_ctxs = get_hard_negative_context(retriever=retriever,
                                                               question=question["question"],
                                                               answer=answers[0],
                                                               n_ctxs=30)
                positive_ctxs = [{
                    "title": f"{article_title}_{i}",
                    "text": c
                } for i, c in enumerate(limit_context_size(question["answers"], context))]

                if not hard_negative_ctxs or not positive_ctxs:
                    logging.error(
                        f"No retrieved candidates for article {article_title}, with question {question['question']}")
                    n_non_added_questions += 1
                    continue
                dict_DPR = {
                    "question": question["question"],
                    "answers": answers,
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": [],
                    "hard_negative_ctxs": hard_negative_ctxs
                }
                list_DPR.append(dict_DPR)
                n_questions += 1

        if idx_article % int(len(squad_data) * 0.5) == 0:
            yield list_DPR
            list_DPR.clear()
            # list_DPR.clear()

    print(f"Number of not added questions : {n_non_added_questions}")
def save_complete_dataset(iter_dpr: Iterator, dpr_outpupt_path: Path):
    for list_dpr in iter_dpr:
        # list_dpr = chunk
        random.shuffle(list_dpr)

        dataset_file_name = dpr_outpupt_path / Path(f"DPR_FR_all.json")

        if dataset_file_name.exists():
            with open(dataset_file_name) as json_ds:
                saved_data = json.load(json_ds)
            saved_data.extend(list_dpr)
        else:
            saved_data = list_dpr

        with open(dataset_file_name, "w") as json_ds:
            json.dump(saved_data, json_ds, indent=4)
    return dataset_file_name


def get_hard_negative_context(retriever: ElasticsearchRetriever, question: str, answer: str,
                              n_ctxs: int = 30, n_chars: int = 600):
    list_hard_neg_ctxs = []
    retrieved_docs = retriever.retrieve(query=question, top_k=n_ctxs, index="document")
    for retrieved_doc in retrieved_docs:
        retrieved_doc_id = retrieved_doc.meta["name"]
        retrieved_doc_text = retrieved_doc.text
        if answer.lower() in retrieved_doc_text.lower():
            continue
        list_hard_neg_ctxs.append({"title": retrieved_doc_id, "text": retrieved_doc_text[:n_chars]})

    return list_hard_neg_ctxs


def split_dataset(dataset_file_name: Path, training_proportion: float = 0.8):
    with open(dataset_file_name.as_posix()) as ds:
        dataset_json = json.load(ds)
    nb_train_sample = int(len(dataset_json) * training_proportion)
    train_file_name = dataset_file_name.parent / Path(dataset_file_name.stem + "_train.json")
    dev_file_name = dataset_file_name.parent / Path(dataset_file_name.stem + "_dev.json")

    train = dataset_json[:nb_train_sample]
    dev = dataset_json[nb_train_sample:]

    train_questions = set([q["question"] for q in train])
    dev = [q for q in dev if q["question"] not in train_questions]

    dataset_complete = {"train": (train, train_file_name),
                        "dev": (dev, dev_file_name)}

    for _, (dataset, dataset_path) in dataset_complete.items():
        with open(dataset_path, "w") as filo:
            json.dump(dataset, filo, indent=4, ensure_ascii=False)


def main(squad_file_path: Path, dpr_output_path: Path):
    tqdm.write(f"Using SQuAD-like file {squad_file_path}")
    list_DPR = create_dpr_training_dataset(squad_file_path)

    dataset_file_name = save_complete_dataset(iter_dpr=list_DPR, dpr_outpupt_path=dpr_output_path)
    split_dataset(dataset_file_name=dataset_file_name)
    return


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    squad_file_path = Path(parser.squad_file_path)
    dpr_output_path = Path(parser.dpr_output_path)
    main(squad_file_path=squad_file_path, dpr_output_path=dpr_output_path)
