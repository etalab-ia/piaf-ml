# About your dataset: if you don't have annotations attached to your texts, don't forget to add an empty array like this "qas": []
# we suggest you put your SQuAD formatted dataset into the /data folder
# make sure your dataset has "root" owner

from pathlib import Path

from rest_api.config import PIPELINE_YAML_PATH, QUERY_PIPELINE_NAME
from haystack import Pipeline
from haystack.preprocessor.preprocessor import PreProcessor
# THIS IMPORT IS NEEDED: the Pipeline.load_from_yaml will not see the TitleEmbeddingRetriever
from rest_api.pipeline.custom_component import TitleEmbeddingRetriever

evaluation_data = Path("./data/squad.json")
split_by = "word"
split_length = 1000

ES_host = "elasticsearch"

import logging

from elasticsearch import Elasticsearch

port = "9200"


def delete_indices(index="document"):
    logging.info(f"Delete index {index} inside Elasticsearch ...")
    es = Elasticsearch([f"http://{ES_host}:{port}/"], verify_certs=True)
    es.indices.delete(index=index, ignore=[400, 404])


doc_index = "document_elasticsearch"
label_index = "label_elasticsearch"

PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)

preprocessor = PreProcessor(
    clean_empty_lines=False,
    clean_whitespace=False,
    clean_header_footer=False,
    split_by=split_by,
    split_length=split_length,
    split_overlap=0,
    split_respect_sentence_boundary=False,
)

es_retriever = PIPELINE.get_node("ESRetriever")
title_emb_retriever = PIPELINE.get_node("TitleEmbRetriever")
document_store = es_retriever.document_store
delete_indices(index=doc_index)
document_store.delete_documents(index=doc_index)
document_store.delete_documents(index=label_index)
document_store.add_eval_data(
    evaluation_data.as_posix(),
    doc_index=doc_index,
    label_index=label_index,
    preprocessor=preprocessor,
)
document_store.update_embeddings(title_emb_retriever, index=doc_index)
