from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers
import pandas as pd
import requests
import logging
import subprocess
import time
from pathlib import Path
import json

from src.util.convert_json_to_dictsAndEmbeddings import convert_json_to_dictsAndEmbeddings

DENSE_MAPPING = {
   "mappings":{
      "properties":{
         "link":{
            "type":"keyword"
         },
         "name":{
            "type":"keyword"
         },
         "text":{
            "type":"text"
         },
         "question_sparse": {
            "type": "text"
         },
         "embedding": {
            "type": "dense_vector",
            "dims": 512
         },
         "audience":{
            "type":"keyword"
         },
         "theme":{
            "type":"keyword"
         },
         "sous_theme":{
            "type":"keyword"
         },
         "dossier":{
            "type":"keyword"
         },
         "sous_dossier":{
            "type":"keyword"
         }
      }
   }
}


document_store = ElasticsearchDocumentStore(host="haystack_elasticsearch_1", username="", password="", index="document",
                                            embedding_field="question_emb", embedding_dim=512,
                                            excluded_meta_data=["question_emb"], custom_mapping=DENSE_MAPPING)
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="distiluse-base-multilingual-cased",
                               use_gpu=False, model_format="sentence_transformers")




dicts = convert_json_to_dictsAndEmbeddings(dir_path="./data/v12", retriever=retriever,
                                           split_paragraphs=False)
# import pickle
# pickle.dump(dicts, open("./data/v11_dicts.pkl", "wb"))

document_store.write_documents(dicts)


