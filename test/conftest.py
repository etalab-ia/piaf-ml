import os
import subprocess
import sys
import time

import pytest
import torch
from elasticsearch import Elasticsearch
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.retriever.sparse import ElasticsearchRetriever

sys.path.insert(0, os.path.abspath("./"))
from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING


@pytest.fixture(scope="session", autouse=True)
def elasticsearch_fixture():
    # test if a ES cluster is already running. If not, download and start an ES instance locally.
    try:
        client = Elasticsearch(hosts=[{"host": "localhost", "port": "9200"}])
        client.info()
    except:
        print("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker rm haystack_test_elastic'],
            shell=True
        )
        status = subprocess.run(
            ['docker run -d --name haystack_test_elastic -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'],
            shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Elasticsearch. Please check docker container logs.")
        time.sleep(30)



@pytest.fixture
def gpu_available():
    return torch.cuda.is_available()


@pytest.fixture(scope='session')
def document_store(elasticsearch_fixture):
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                create_index=False, embedding_field="emb",
                                                embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                custom_mapping=SQUAD_MAPPING)
    yield document_store
    document_store.delete_all_documents(index='document')


@pytest.fixture
def retriever_bm25(document_store):
    return ElasticsearchRetriever(document_store=document_store)


@pytest.fixture
def retriever_emb(document_store, gpu_available):
    return EmbeddingRetriever(document_store=document_store,
                              embedding_model="distiluse-base-multilingual-cased",
                              use_gpu=gpu_available, model_format="sentence_transformers",
                              pooling_strategy="reduce_max",
                              emb_extraction_layer=-1)
