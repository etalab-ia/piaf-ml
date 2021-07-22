import logging
import json
from random import randint, choice
from pathlib import Path
from typing import Optional, List

from haystack import Document

from haystack.retriever.base import BaseRetriever
from haystack.document_store.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class EpitcaRetriever(BaseRetriever):

    def __init__(self,
                 document_store: BaseDocumentStore,
                 top_k: int = 10,
                 epitca_perf_file: str = "clients/cnil/knowledge_base/raw_and_preparation/epitca_perf_V2.json"):
        """
            The EpitcaRetriever is intended for simulating the CNIL search
            engine Epitca. It retrieves the documents listed as answers to the
            questions in `epitca_perf_file`.

            :param document_store: The document store that is used for
            retrieving documents. Each document should contain a field "id" that
            corresponds to the ids appearing in epitca_perf_file.
            :param top_k: How many documents to retrieve
            :param epitca_perf_file: the path to the json file containing
            Epitca's performance data. It gives a list of questions,  the
            documents ids returned by Epitca as well as the document id for the
            expected response.
        """
        self.document_store = document_store
        self.top_k = top_k
        self.epitca_perf_file = Path(epitca_perf_file)
        self.epitca_perf_dict = load_perf_file(self.epitca_perf_file)

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[
        Document]:
        """
            Scan through documents in DocumentStore and return a small number documents
            that are most relevant to the query.

            :param query: The query
            :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
            :param top_k: How many documents to return per query.
            :param index: The name of the index in the DocumentStore from which to retrieve documents
            :return: a list of Document retrieved
        """

        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve() since initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index

        try:
            result_doc_ids = self.epitca_perf_dict[query][:top_k]
        except KeyError:
            raise KeyError(f"Query '{query}' not found in Epticas performance data")

        documents = self.document_store.query("*",
            filters={"id": result_doc_ids})

        return documents


def load_perf_file(epitca_perf_file):
    """
    This loads the Epitca performance data.

    :returns: a dictionary where the keys are the questions asked and values
    """
    data = {}
    with open(epitca_perf_file, 'r') as f:
        data = json.load(f)

    epitca_perf_dict = {}

    for expected_answer_group in data:
        for question in expected_answer_group["epitca"]:
            epitca_perf_dict[question["question"]] = \
                    [result["document"]["id"] for result in question["result"]]

    return epitca_perf_dict


def load_perf_file_expected_answer(epitca_perf_file):
    """
    This loads the Epitca performance data.

    :returns: a dictionary where the keys are the questions asked and values
    """
    data = {}
    with open(epitca_perf_file, 'r') as f:
        data = json.load(f)

    expected = {}

    for expected_answer_group in data:
        for question in expected_answer_group["epitca"]:
            expected[question["question"]] = \
                    str(expected_answer_group["expected"])

    return expected
