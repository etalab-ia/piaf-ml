import logging
import json
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


MAPPING = {
    "mappings": {
        "properties": {
            "link": {"type": "keyword"},
            "name": {"type": "keyword"},
            "text": {"type": "text"},
            "question_sparse": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 512},
            "audience": {"type": "keyword"},
            "theme": {"type": "keyword"},
            "sous_theme": {"type": "keyword"},
            "dossier": {"type": "keyword"},
            "sous_dossier": {"type": "keyword"},
        }
    },
    "settings": ANALYZER_DEFAULT,
}

document_store = ElasticsearchDocumentStore(
    host="haystack_elasticsearch_1",
    username="",
    password="",
    index="document_elasticsearch",
    custom_mapping=MAPPING,
)

# New way of getting docs from SQuAD like format
from haystack.preprocessor.utils import eval_data_from_json

dicts, labels = eval_data_from_json("data/squad_lie_formated.json")
document_store.write_documents(dicts)

# Old way - will soon be deleted (as of mar 2021, making the change) - for folders filled with .json files
# def get_arbo(dict, level):
#     if "arborescence" in dict:
#         try:
#             return dict["arborescence"][level]
#         except:  # in case the level does not exist or there is no dict at all
#             return ''


# def convert_json_files_to_dicts(dir_path: str):
#     file_paths = [p for p in Path(dir_path).glob("**/*")]
#     documents = []

#     for path in tqdm(file_paths):
#         if path.suffix.lower() == ".json":
#             with open(path) as doc:
#                 json_doc = json.load(doc)

#             text = json_doc["text"]
#             embedding = []
#             audience = get_arbo(json_doc, 'audience')
#             theme = get_arbo(json_doc, 'theme')
#             sous_theme = get_arbo(json_doc, 'sous_theme')
#             dossier = get_arbo(json_doc, 'dossier')
#             sous_dossier = get_arbo(json_doc, 'sous_dossier')
#         else:
#             raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

#         text_reader = json_doc["text_reader"] if "text_reader" in json_doc else text
#         # we have to remove "embedding": embedding from the JSON otherwise if will raise an error
#         documents.append({"text": text_reader,
#                           "question_sparse": text,
#                           "meta": {"name": path.name,
#                                    "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
#                                    "audience": audience,
#                                    "theme": theme,
#                                    "sous_theme": sous_theme,
#                                    "dossier": dossier,
#                                    "sous_dossier": sous_dossier}})
#     return documents


# dicts = convert_json_files_to_dicts(dir_path="data/v14")
# document_store.write_documents(dicts)
