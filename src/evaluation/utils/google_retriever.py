import logging
from random import randint, choice
from time import sleep
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
                 website: str = "service-public.fr"):
        """
            The GoogleRetriever is intended for establishing a baseline of the final clients' expectations.

            :param document_store: The documentstore that is used for retrieving documents
            :param top_k: How many documents to retrieve
            :param website: The website in which google has to find results. The url retrieved by google will then be search inside the document_store
        """

        self.document_store = document_store
        self.top_k = top_k
        self.website = website
        self.headers = [
            {
                'authority': 'www.google.com',
                'cache-control': 'max-age=0',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Mobile Safari/537.36',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'sec-fetch-site': 'none',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-user': '?1',
                'sec-fetch-dest': 'document',
                'accept-language': 'fr-FR,fr;q=0.9',
            },
            {
                'authority': 'www.google.com',
                'cache-control': 'max-age=0',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Android 4.4; Tablet; rv:70.0) Gecko/70.0 Firefox/70.0',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'service-worker-navigation-preload': '0rPUmQUGCAAQABgA',
                'sec-fetch-site': 'none',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-user': '?1',
                'sec-fetch-dest': 'document',
                'accept-language': 'fr-FR,fr;q=0.9'},
            {
                'authority': 'www.google.com',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'service-worker-navigation-preload': '0rPUmQUGCAAQABgA',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-user': '?1',
                'sec-fetch-dest': 'document',
                'referer': 'https://www.google.com/',
                'accept-language': 'fr-FR,fr;q=0.9'}
        ]
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Mozilla/5.0 (Android 4.4; Tablet; rv:70.0) Gecko/70.0 Firefox/70.0',
            'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Mobile Safari/537.36'
        ]

    @backoff.on_exception(backoff.expo, HTTPError, max_time=120)
    def get_gsearch_results(self, query, top_k):
        rand_sleep = randint(60, 300)
        logger.info(f"Search then sleep for {rand_sleep}s")
        return [url for url in search(query, tld="fr", num=top_k, stop=top_k,
                                      pause=rand_sleep, user_agent=choice(self.user_agents))]

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[
        Document]:
        """
            Scan through documents in DocumentStore and return a small number documents
            that are most relevant to the query.

            :param query: The query
            :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
            :param top_k: How many documents to return per query.
            :param index: The name of the index in the DocumentStore from which to retrieve documents
        """

        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index

        query_website = f"site:{self.website} " + query  # add website restriction to query

        gsearch_results = self.get_gsearch_results(query_website, top_k)  # the list of plain url retrieved by google

        # TODO sometimes the doc is not found we should still have top_k results
        documents = []
        for g in gsearch_results:
            document_list = self.document_store.query("*", filters={"link": [g]})
            if len(document_list) > 0:
                documents.append(document_list[0])  # used for avoid http error 429 too many request
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
