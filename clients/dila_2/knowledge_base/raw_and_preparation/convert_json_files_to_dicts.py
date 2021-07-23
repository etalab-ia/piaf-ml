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
            raise Exception(
                f"Indexing of {path.suffix} files is not currently supported."
            )

        documents.append(
            {
                "text": text,
                "meta": {
                    "name": path.name,
                    "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
                },
            }
        )

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

                def get_arbo(dict, level):
                    try:
                        return dict[level]
                    except:  # in case the level does not exist or there is no dict at all
                        return "N.A"

                jsonDoc = json.load(doc)
                text = jsonDoc["text"]
                arborescence = jsonDoc["arborescence"]
                audience = get_arbo(arborescence, "audience")
                theme = get_arbo(arborescence, "theme")
                sous_theme = get_arbo(arborescence, "sous_theme")
                dossier = get_arbo(arborescence, "dossier")
                sous_dossier = get_arbo(arborescence, "sous_dossier")
        else:
            raise Exception(
                f"Indexing of {path.suffix} files is not currently supported."
            )

        documents.append(
            {
                "text": text,
                "meta": {
                    "name": path.name,
                    "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
                    "audience": audience,
                    "theme": theme,
                    "sous_theme": sous_theme,
                    "dossier": dossier,
                    "sous_dossier": sous_dossier,
                },
            }
        )

    return documents
