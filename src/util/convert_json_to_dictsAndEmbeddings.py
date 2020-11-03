import json
from pathlib import Path
from typing import Optional, Callable, Union

from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from tqdm import tqdm
from haystack import Document
import spacy

# if not installed python -m spacy download fr_core_news_sm
try:
    NLP = spacy.load('fr_core_news_sm', disable=['ner', 'parser'])
except:
    NLP = None


def preprocess_text(text: str):
    """
    Tokenize, lemmatize, lowercase and remove stop words
    :param text:
    :return:
    """
    if not NLP:
        print('Warning NLP not loaded, text will not be preprocessed')
        return text

    doc = NLP(text)
    text = " ".join(t.lemma_.lower() for t in doc if not t.is_stop).replace("\n", " ")
    return text

def no_preprocessing(text: str):
    return text


def convert_json_to_dicts(dir_path: str,
                          retriever: Union[EmbeddingRetriever, DensePassageRetriever,
                                           ElasticsearchRetriever],
                          clean_func: Optional[Callable] = None,
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

    for i, path in tqdm(enumerate(file_paths[:])):
        if path.suffix.lower() == ".json":
            with open(path) as doc:
                json_doc = json.load(doc)
            id = i
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
        if clean_func:
            text = clean_func(text)

        # TODO: evaluate performances based on text_reader or text in 'text'
        doc_dict = {"text": text_reader,
                    'question_sparse': text,
                    'embedding': embedding,
                    "meta": {"name": path.name,
                             'id_doc': str(id),
                             "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
                             'audience': audience,
                             'theme': theme,
                             'sous_theme': sous_theme,
                             'dossier': dossier,
                             'sous_dossier': sous_dossier}}
        documents.append(doc_dict)

    return documents
