import pytest
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING


@pytest.fixture
def document_store():
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                create_index=False, embedding_field="emb",
                                                embedding_dim=512, excluded_meta_data=["emb"], similarity='cosine',
                                                custom_mapping=SQUAD_MAPPING)
    return document_store