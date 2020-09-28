import json
from pathlib import Path
from typing import List


def convert_json_files_to_dicts(dir_path: str) -> List[dict]:
    """
    Convert all Json in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.
    expected format for the input Jsons is :
    {
        text : string,
        link : string
    }
    :param dir_path: path for the documents to be written to the database

    :return: [dict]
    """

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

def convert_json_files_v10_to_dicts(dir_path: str) -> List[dict]:
    """
    Convert all Json in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.
    expected format for the input Jsons is :
    {
        text : string,
        link : string,
        arborescence : dict
    }
    :param dir_path: path for the documents to be written to the database

    :return: [dict]
    """

    file_paths = [p for p in Path(dir_path).glob("**/*")]

    documents = []
    for path in file_paths:
        if path.suffix.lower() == ".json":
            with open(path) as doc:
                jsonDoc = json.load(doc)
                text = jsonDoc["text"]
                arborescence = jsonDoc['arborescence']
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

        documents.append({"text": text, "meta": {"name": path.name,
                                                 "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
                                                 'arborescence': arborescence}})

    return documents
