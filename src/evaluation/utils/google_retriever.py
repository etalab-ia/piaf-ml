import logging
import json
from random import randint, choice
from pathlib import Path
from typing import Optional, List

from googlesearch import search
import backoff
from urllib.error import HTTPError
from haystack import Document

from haystack.retriever.base import BaseRetriever
from haystack.document_store.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class GoogleRetriever(BaseRetriever):

    def __init__(self,
                 document_store: BaseDocumentStore,
                 top_k: int = 10,
                 website: str = None,
                 retrieved_search_file: str = "./data/retrieved_search.json"):
        """
            The GoogleRetriever is intended for establishing a baseline of the final clients' expectations.
            It launches a search with google on an optional specific website and retrieves the links found by google.
            It then searches in the database when the weblink for this document is available in the field 'link'.

            :param document_store: The documentstore that is used for retrieving documents
            :param top_k: How many documents to retrieve
            :param website: The website in which google has to find results. The url retrieved by google will then be search inside the document_store
            :param retrieved_search_file: the path to the backup of the links retrieved by google. This must be a json
        """

        self.document_store = document_store
        self.top_k = top_k
        self.website = website
        self.retrieved_search_file = Path(retrieved_search_file)
        self.retrieved_search = self.load_retrieved_search()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Mozilla/5.0 (Android 4.4; Tablet; rv:70.0) Gecko/70.0 Firefox/70.0',
            'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Mobile Safari/537.36'
        ]

    def load_retrieved_search(self):
        """
        This loads the previously retrieved results that are saved in the json file retrieved_search_file
        :return: the data previously retrieved
        """
        data = {}
        if self.retrieved_search_file.exists():
            with open(self.retrieved_search_file, 'r') as f:
                data = json.load(f)
        return data

    def dump_retrieved_search(self):
        """
        This dumps the retrieved results in the json file retrieved_search_file
        """
        with open(self.retrieved_search_file, 'w') as f:
            json.dump(self.retrieved_search, f)


    @backoff.on_exception(backoff.expo, HTTPError, max_time=120)
    def get_gsearch_results(self, query, top_k):
        """
        This function is used to retrieve the results from google.
        To prevent the ban from google, it randomly picks a user agent in self.user_agents.
        It also adds a random sleep of 1 to 3 minutes between requests.
        :param query: the query to ask google
        :param top_k: the number of answers to find
        :return: a list of the urls of the google search results
        """
        rand_sleep = randint(60, 300)
        logger.info(f"Search then sleep for {rand_sleep}s")
        retrieve_search = [url for url in search(query, tld="fr", num=top_k, stop=top_k,
                                      pause=rand_sleep, user_agent=choice(self.user_agents))]
        self.retrieved_search[query] = retrieve_search
        self.dump_retrieved_search()
        return retrieve_search

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
                "Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index

        if self.website:
            query_website = f"site:{self.website} " + query  # add website restriction to query
        else:
            query_website = query

        if query_website not in self.retrieved_search.keys():# Document not previously retrieved
            logger.info(f"Query never performed, let me google that for you !")
            gsearch_results = self.get_gsearch_results(query_website, top_k)  # the list of plain url retrieved by google
        else:
            logger.info("Let's look in previously found results")
            gsearch_results = self.retrieved_search[query_website]
            if len(gsearch_results) <= top_k:
                gsearch_results = gsearch_results[0:top_k]
            else:
                logger.info(f"We did not previously gather enough results, googling again")
                gsearch_results = self.get_gsearch_results(query_website, top_k)

        documents = []
        for g in gsearch_results:
            document_list = self.document_store.query("*", filters={"link": [g]})
            if len(document_list) > 0:
                documents.append(document_list[0])
        return documents


if __name__ == "__main__":
    from pathlib import Path
    from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
    from src.evaluation.config.elasticsearch_mappings import SQUAD_MAPPING
    from src.evaluation.utils.elasticsearch_management import delete_indices, launch_ES

    launch_ES()

    # indexes for the elastic search
    doc_index = "document_xp"
    label_index = "label_xp"

    # deleted indice for elastic search to make sure mappings are properly passed
    delete_indices(index=doc_index)
    delete_indices(index=label_index)

    document_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index=doc_index,
        search_fields=["name", "text"],
        create_index=False,
        embedding_field="emb",
        scheme="",
        embedding_dim=768,
        excluded_meta_data=["emb"],
        similarity="cosine",
        custom_mapping=SQUAD_MAPPING,
    )

    # Add evaluation data to Elasticsearch document store
    evaluation_data = Path("./clients/dila/knowledge_base/squad.json")
    document_store.add_eval_data(
        evaluation_data.as_posix(),
        doc_index=doc_index,
        label_index=label_index,
    )

    website = 'service-public.fr'
    query = "'Puis-je choisir un fournisseur de gaz différent du fournisseur d\'électricité ?'"
    query_website = f"site:{website} " + query

    gsearch_results = [url for url in search(query_website, tld="fr", num=10, stop=10, pause=2)]

    documents = []
    for g in gsearch_results:
        document_list = document_store.query("*", filters={"link": [g]})
        if len(document_list) > 0:
            documents.append(document_list[0])
