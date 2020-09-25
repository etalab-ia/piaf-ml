import logging
import json
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from pathlib import Path

logger = logging.getLogger(__name__)

document_store = ElasticsearchDocumentStore(host="haystack2_elasticsearch_1", username="", password="", index="document")

def convert_json_files_to_dicts(dir_path: str):
    file_paths = [p for p in Path(dir_path).glob("**/*")]
    documents = []
    for path in file_paths:
        if path.suffix.lower() == ".json":
            with open(path) as doc:
                jsonDoc = json.load(doc)
                text = jsonDoc["text"]
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

        documents.append({"text": text, "meta": {"name": path.name,
                                                 "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}"}})

    return documents

dicts = convert_json_files_to_dicts(dir_path="data/v8")
document_store.write_documents(dicts)
