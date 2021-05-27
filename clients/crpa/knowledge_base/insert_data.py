# About your dataset: if you don't have annotations attached to your texts, don't forget to add an empty array like this "qas": []
# we suggest you put your SQuAD formatted dataset into the /data folder
# make sure your dataset has "root" owner

import hashlib
import torch
import socket
import time
from datetime import datetime

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


evaluation_data = Path("./data/dataset.json")
k_retriever = 20
k_title_retriever = 10
k_reader_per_candidate = 5
k_reader_total = 3
preprocessing = True
split_by = "word"
split_length = 1000
title_boosting_factor = 1

ES_host = "haystack_crpa_elasticsearch_1"

from typing import List
import numpy as np
from haystack import Document


class TitleEmbeddingRetriever(EmbeddingRetriever):
    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings of the titles for a list of passages. For this Retriever type: The same as calling .embed()
        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        texts = [d.meta["name"] for d in docs]

        return self.embed(texts)


import subprocess
import platform
import logging

from elasticsearch import Elasticsearch

port = "9200"


def launch_ES():
    logging.info("Search for Elasticsearch ...")
    es = Elasticsearch([f"http://{ES_host}:{port}/"], verify_certs=True)
    if not es.ping():
        logging.info("Elasticsearch not found !")
        logging.info("Starting Elasticsearch ...")
        if platform.system() == "Windows":
            status = subprocess.run(
                f'docker run -d -p {port}:{port} -e "discovery.type=single-node" elasticsearch:7.6.2'
            )
        else:
            status = subprocess.run(
                [
                    f'docker run -d -p {port}:{port} -e "discovery.type=single-node" elasticsearch:7.6.2'
                ],
                shell=True,
            )
        time.sleep(30)
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch.")
    else:
        logging.info("Elasticsearch found !")


def delete_indices(index="document"):
    logging.info(f"Delete index {index} inside Elasticsearch ...")
    es = Elasticsearch([f"http://{ES_host}:{port}/"], verify_certs=True)
    es.indices.delete(index=index, ignore=[400, 404])


def prepare_mapping(
    mapping, preprocessing, title_boosting_factor=1, embedding_dimension=512
):
    mapping["mappings"]["properties"]["name"]["boost"] = title_boosting_factor
    mapping["mappings"]["properties"]["emb"]["dims"] = embedding_dimension
    if not preprocessing:
        mapping["settings"] = {
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard",
                    }
                }
            }
        }


ANALYZER_DEFAULT = {
    "analysis": {
        "filter": {
            "french_elision": {
                "type": "elision",
                "articles_case": True,
                "articles": [
                    "l",
                    "m",
                    "t",
                    "qu",
                    "n",
                    "s",
                    "j",
                    "d",
                    "c",
                    "jusqu",
                    "quoiqu",
                    "lorsqu",
                    "puisqu",
                ],
            },
            "french_stop": {"type": "stop", "stopwords": "_french_"},
            "french_stemmer": {"type": "stemmer", "language": "light_french"},
        },
        "analyzer": {
            "default": {
                "tokenizer": "standard",
                "filter": [
                    "french_elision",
                    "lowercase",
                    "french_stop",
                    "french_stemmer",
                ],
            }
        },
    }
}

SQUAD_MAPPING = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "text": {"type": "text"},
            "emb": {"type": "dense_vector", "dims": 512},
        },
        "dynamic_templates": [
            {
                "strings": {
                    "path_match": "*",
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }
            }
        ],
    },
    "settings": ANALYZER_DEFAULT,
}


doc_index = "document_elasticsearch"
label_index = "label_elasticsearch"

delete_indices(index=doc_index)
prepare_mapping(
    mapping=SQUAD_MAPPING,
    preprocessing=preprocessing,
    title_boosting_factor=title_boosting_factor,
    embedding_dimension=512,
)

preprocessor = PreProcessor(
    clean_empty_lines=False,
    clean_whitespace=False,
    clean_header_footer=False,
    split_by=split_by,
    split_length=split_length,
    split_overlap=0,
    split_respect_sentence_boundary=False,
)

document_store = ElasticsearchDocumentStore(
    host="haystack_crpa_elasticsearch_1",
    username="",
    password="",
    index=doc_index,
    search_fields=["name", "text"],
    create_index=False,
    embedding_field="emb",
    embedding_dim=512,
    excluded_meta_data=["emb"],
    similarity="cosine",
    custom_mapping=SQUAD_MAPPING,
)
retriever = TitleEmbeddingRetriever(
    document_store=document_store,
    embedding_model="distiluse-base-multilingual-cased",
    use_gpu=False,
    model_format="sentence_transformers",
    pooling_strategy="reduce_max",
    emb_extraction_layer=-1,
)

document_store.delete_all_documents(index=doc_index)
document_store.delete_all_documents(index=label_index)
document_store.add_eval_data(
    evaluation_data.as_posix(),
    doc_index=doc_index,
    label_index=label_index,
    preprocessor=preprocessor,
)
document_store.update_embeddings(retriever, index=doc_index)
