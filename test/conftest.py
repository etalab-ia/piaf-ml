import os
import sys

import pytest
import torch
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.retriever.sparse import ElasticsearchRetriever

sys.path.insert(0, os.path.abspath("./"))
from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING


@pytest.fixture
def gpu_available():
    return torch.cuda.is_available()


@pytest.fixture(scope='session')
def document_store():
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
                              emb_extraction_layer=-2)
