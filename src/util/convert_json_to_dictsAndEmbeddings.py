import json
from pathlib import Path
from typing import Optional, Callable, Union

from haystack import Document
from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from tqdm import tqdm
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

    for path in tqdm(file_paths[:]):
        if path.suffix.lower() == ".json":
            with open(path, encoding='utf-8') as doc:
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
        if clean_func:
            text = clean_func(text)
        # TODO: evaluate performances based on text_reader or text in 'text'
        doc_dict = {"text": text_reader,
                    'question_sparse': text,
                    'question_emb': embedding,
                    "meta": {"name": path.name,
                             "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}",
                             'audience': audience,
                             'theme': theme,
                             'sous_theme': sous_theme,
                             'dossier': dossier,
                             'sous_dossier': sous_dossier}}
        # ES injection fails if embedding is [] while dim = 512, so we need to remove the whole porp
        if not compute_embeddings:
            doc_dict.pop('question_emb')
        documents.append(doc_dict)

    return documents

def convert_squad_to_dicts(dir_path: str,
                          retriever: Union[EmbeddingRetriever, DensePassageRetriever,
                                           ElasticsearchRetriever],
                          clean_func: Optional[Callable] = None,
                          split_paragraphs: bool = False,
                          compute_embeddings: bool = False):
    """
    Convert the contexts of the squad dataset of the given path to Python dicts that can be written to a
    Document Store.

    :param compute_embeddings:
    :param retriever:
    :param squad_path: path for the squad dataset to be written to the database
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    :return: None
    """
    if split_paragraphs:
        raise Exception(f"Splitting paragraph not currently supported.")

    with open(dir_path, encoding='UTF-8') as f:
        squad = json.load(f)

    documents = []

    list_articles = squad['data']
    for document in list_articles:
        # get all extra fields from document level (e.g. title)
        meta_doc = {k: v for k, v in document.items() if k not in ("paragraphs", "title")}
        for paragraph in document['paragraphs']:
            cur_meta = {"name": document.get("title", None)}
            # all other fields from paragraph level
            meta_paragraph = {k: v for k, v in paragraph.items() if k not in ("qas", "context")}
            cur_meta.update(meta_paragraph)
            # meta from parent document
            cur_meta.update(meta_doc)

            text = paragraph['context']
            embedding = []
            if compute_embeddings:
                assert retriever is not None
                embedding = retriever.embed_passages(docs=[Document(text=text)])[0]

            text_reader = paragraph["text_reader"] if "text_reader" in paragraph else text
            if clean_func:
                text = clean_func(text)

            doc_dict = {"text": text_reader,
                        'question_sparse': text,
                        'question_emb': embedding,
                        'meta': cur_meta}
            # ES injection fails if embedding is [] while dim = 512, so we need to remove the whole porp
            if not compute_embeddings:
                doc_dict.pop('question_emb')
            documents.append(doc_dict)

    return documents
