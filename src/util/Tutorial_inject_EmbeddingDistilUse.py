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

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document", embedding_field="question_emb", embedding_dim=512, excluded_meta_data=["question_emb"])
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="distiluse-base-multilingual-cased", use_gpu=False, model_format="sentence_transformers" )

def convert_json_to_dictsAndEmbeddings(dir_path: str, split_paragraphs: bool = False):
    file_paths = [p for p in Path(dir_path).glob("**/*")]

    documents = []
    for path in file_paths:
        if path.suffix.lower() == ".json":
            with open(path) as doc:
                jsonDoc = json.load(doc)
                text = jsonDoc["text"]
                question_emb = retriever.embed(texts=text)[0]
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

        if split_paragraphs:
            raise Exception(f"Splitting paragraph not currently supported.")
        else:
            documents.append({"text": text, "question": text, "question_emb": question_emb, "meta": {"name": path.name, "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}" }})
    return documents


dicts = convert_json_to_dictsAndEmbeddings(dir_path="data/v8", split_paragraphs=False)
document_store.write_documents(dicts)
