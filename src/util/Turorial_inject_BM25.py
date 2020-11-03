import logging
import json
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

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

document_store = ElasticsearchDocumentStore(host="haystack2_elasticsearch_1", username="", password="", index="document_elasticsearch", custom_mapping=DENSE_MAPPING)


def get_arbo(dict, level):
    if "arborescence" in dict:
        try:
            return dict["arborescence"][level]
        except:  # in case the level does not exist or there is no dict at all
            return ''


def convert_json_files_to_dicts(dir_path: str):
    file_paths = [p for p in Path(dir_path).glob("**/*")]
    documents = []
    
    for path in tqdm(file_paths):
        if path.suffix.lower() == ".json":
            with open(path) as doc:
                json_doc = json.load(doc)

            text = json_doc["text"]
            audience = get_arbo(json_doc, 'audience')
            theme = get_arbo(json_doc, 'theme')
            sous_theme = get_arbo(json_doc, 'sous_theme')
            dossier = get_arbo(json_doc, 'dossier')
            sous_dossier = get_arbo(json_doc, 'sous_dossier')
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

        text_reader = json_doc["text_reader"] if "text_reader" in json_doc else text
        documents.append({"text": text,
                          "meta": {"name": path.name,
                                   "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
                                   "audience": audience,
                                   "theme": theme,
                                   "sous_theme": sous_theme,
                                   "dossier": dossier,
                                   "sous_dossier": sous_dossier}})
    return documents
    
    

dicts = convert_json_files_to_dicts(dir_path="data/v11")
document_store.write_documents(dicts)
