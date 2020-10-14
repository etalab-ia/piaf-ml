"""
Script to convert a SQuAD-like QA-dataset format JSON file to DPR Dense Retriever training format

Usage:
    SQuAD2DPR_format.py <squad_file_path> <dpr_output_path> [options]

Arguments:
    <squad_file_path>   SQuAD file path
    <dpr_output_path>   DPR outpput file path
"""
import logging
import subprocess
from pathlib import Path
from time import sleep
from typing import List, Dict

from argopt import argopt
from elasticsearch import Elasticsearch
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from tqdm import tqdm
import json
import random
import re

random.seed(42)

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


def limit_context_size(answers_list, context: str, max_char_size=600):
    """
    Takes a context string and limits its size to around 100 words per context.
    As I do not want to deal for now with spans and tokenization and machin, I will assume
    a word is 5 chars in average. So a context will have max 500 size by default.
    It considers the answer (of length q), and grabs the n previous characters and the m next characters so that
    q+m+n =~ max_char_size

    :param max_char_size: Max size of the tet chink (in chars)
    :param answers_list: list containing the answers' infos [{"answer_start" : "...", "text" : "..."}]
    :param context: The wikipedia paragraph where the answer is found
    :return: A list of limited size contexts
    """
    seen_answers = []
    limited_contexts = []
    for answer in answers_list:
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


def get_hard_negative_context(retriever: ElasticsearchRetriever, question: str, answer: str):
    retriever_docs = retriever.retrieve(query=question, top_k=5, index="document")
    for retrieved_doc in retriever_docs:
        retrieved_doc_id = retrieved_doc.meta["name"]
        retrieved_doc_text = retrieved_doc.text
        if answer.lower() not in retrieved_doc_text.lower():
            return {"title": retrieved_doc_id, "text": retrieved_doc_text}


def create_dpr_training_dataset(squad_file_path: Path, dpr_output_path: Path):
    squad_file = json.load(open(squad_file_path.as_posix()))
    version = squad_file["version"]
    squad_data = squad_file["data"]
    retriever = prepare_es_retrieval(squad_data=squad_data)
    random.shuffle(squad_data)
    list_DPR = []
    for article in tqdm(squad_data[:1000]):
        article_title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                limited_positive_ctxs = limit_context_size(question["answers"], context)
                answers = [a["text"] for a in question["answers"]]
                dict_DPR = {
                    "question": question["question"],
                    "answers": answers,
                    "positive_ctxs": [{
                        "title": f"{article_title}_{i}",
                        "text": c
                    } for i, c in enumerate(limited_positive_ctxs)],
                    "negative_ctxs": [],
                    "hard_negative_ctxs": [get_hard_negative_context(retriever=retriever,
                                                                question=question["question"],
                                                                answer=answers[0])]
                }
                list_DPR.append(dict_DPR)
    with open(dpr_output_path.as_posix(), "w") as filo:
        json.dump(list_DPR, filo, ensure_ascii=False, indent=4)
    return 1


def main(squad_file_path: Path, dpr_output_path: Path):
    job_output = []
    tqdm.write(f"Using SQuAD-like file {squad_file_path}")
    job_output.append(create_dpr_training_dataset(squad_file_path, dpr_output_path))

    logging.info(
        f"{sum(job_output)} questions in the passed SQuAD-like file passed were added to the DPR Retriever training set"
        f" {len(job_output) - sum(job_output)} files had some error.")

    return


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    squad_file_path = Path(parser.squad_file_path)
    dpr_output_path = Path(parser.dpr_output_path)
    main(squad_file_path=squad_file_path, dpr_output_path=dpr_output_path)
