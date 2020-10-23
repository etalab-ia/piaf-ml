import json
from pathlib import Path

from haystack.retriever.base import BaseRetriever
from haystack.retriever.dense import EmbeddingRetriever
from tqdm import tqdm
from haystack import Document


def convert_json_to_dicts(dir_path: str, retriever: BaseRetriever,
                          split_paragraphs: bool = False,
                          compute_embeddings: bool = False):
    """
    Convert all Json in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.
    expected format for Json is :
    {
        text : string,
        link : string
    }
    :param compute_embeddings:
    :param retriever:
    :param dir_path: path for the documents to be written to the database
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    :return: None
    """

    file_paths = [p for p in Path(dir_path).glob("**/*")]

    documents = []

    def get_arbo(dict, level):
        if "arborescence" in dict:
            try:
                return dict["arborescence"][level]
            except:  # in case the level does not exist or there is no dict at all
                return ''

    for path in tqdm(file_paths[:200]):
        if path.suffix.lower() == ".json":
            with open(path) as doc:
                json_doc = json.load(doc)

            text = json_doc["text"]
            audience = get_arbo(json_doc, 'audience')
            theme = get_arbo(json_doc, 'theme')
            sous_theme = get_arbo(json_doc, 'sous_theme')
            dossier = get_arbo(json_doc, 'dossier')
            sous_dossier = get_arbo(json_doc, 'sous_dossier')
            embedding = []
            if compute_embeddings:
                assert retriever is not None
                embedding = retriever.embed_passages(docs=[Document(text=text)])[0]
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

        if split_paragraphs:
            raise Exception(f"Splitting paragraph not currently supported.")

        text_reader = json_doc["text_reader"] if "text_reader" in json_doc else text
        # TODO: evaluate performances based on text_reader or text in 'text'
        doc_dict = {"text": text_reader,
                          'question_sparse': text,
                          'embedding': embedding,
                          "meta": {"name": path.name,
                                   "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
                                   'audience': audience,
                                   'theme': theme,
                                   'sous_theme': sous_theme,
                                   'dossier': dossier,
                                   'sous_dossier': sous_dossier}}
        documents.append(doc_dict)

    return documents
