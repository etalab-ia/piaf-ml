import os
import subprocess
import sys
import time

import pytest
import torch
from elasticsearch import Elasticsearch
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.reader.transformers import TransformersReader
from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.preprocessor.preprocessor import PreProcessor


sys.path.insert(0, os.path.abspath("./"))
from src.evaluation.utils.elasticsearch_management import delete_indices, prepare_mapping
from src.evaluation.utils.TitleEmbeddingRetriever import TitleEmbeddingRetriever
from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING


@pytest.fixture(scope="session", autouse=True)
def elasticsearch_fixture():
    # test if a ES cluster is already running. If not, download and start an ES instance locally.
    try:
        client = Elasticsearch(hosts=[{"host": "localhost", "port": "9200"}])
        client.info()
    except:
        print("Starting Elasticsearch ...")
        status = subprocess.run(["docker rm haystack_test_elastic"], shell=True)
        status = subprocess.run(
            [
                'docker run -d --name haystack_test_elastic -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'
            ],
            shell=True,
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Elasticsearch. Please check docker container logs."
            )
        time.sleep(30)


@pytest.fixture
def gpu_available():
    return torch.cuda.is_available()


@pytest.fixture
def gpu_id(gpu_available):
    if gpu_available:
        gpu_id = torch.cuda.current_device()
    else:
        gpu_id = -1
    return gpu_id


@pytest.fixture(scope="session")
def preprocessor():
    # test with preprocessor
    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by="word",
        split_length=50,
        split_overlap=0,  # this must be set to 0 at the data of writting this: 22 01 2021
        split_respect_sentence_boundary=False,
    )
    return preprocessor


@pytest.fixture(scope="session")
def document_store(elasticsearch_fixture):
    prepare_mapping(mapping=SQUAD_MAPPING, embedding_dimension=768)

    document_index = "document"
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=document_index,
                                                create_index=False, embedding_field="emb",
                                                embedding_dim=768, excluded_meta_data=["emb"], similarity='cosine',
                                                custom_mapping=SQUAD_MAPPING)
    yield document_store
    # clean up
    delete_indices(index=document_index)
    delete_indices(index="label")


@pytest.fixture
def reader(gpu_id):
    k_reader = 3
    reader = TransformersReader(
        model_name_or_path="etalab-ia/camembert-base-squadFR-fquad-piaf",
        tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf",
        use_gpu=gpu_id,
        top_k_per_candidate=k_reader,
    )
    return reader


@pytest.fixture
def retriever_bm25(document_store):
    return ElasticsearchRetriever(document_store=document_store)


@pytest.fixture
def retriever_emb(document_store, gpu_available):
    return EmbeddingRetriever(document_store=document_store,
                              embedding_model="distilbert-base-multilingual-cased",
                              model_version="1a01b38498875d45f69b2a6721bf6fe87425da39",
                              use_gpu=gpu_available, model_format="transformers",
                              pooling_strategy="reduce_max",
                              emb_extraction_layer=-1)

@pytest.fixture
def retriever_dpr(document_store, gpu_available):
    return DensePassageRetriever(document_store=document_store,
                                 query_embedding_model="etalab-ia/dpr-question_encoder-fr_qa-camembert",
                                 passage_embedding_model="etalab-ia/dpr-ctx_encoder-fr_qa-camembert",
                                 infer_tokenizer_classes=True,
                                 use_gpu=gpu_available)


@pytest.fixture
def retriever_faq(document_store, gpu_available):
    return TitleEmbeddingRetriever(
        document_store=document_store,
        embedding_model="distiluse-base-multilingual-cased",
        use_gpu=gpu_available,
        model_format="sentence_transformers",
        pooling_strategy="reduce_max",
        emb_extraction_layer=-1,
    )
