from typing import Optional, Dict, List
from copy import deepcopy

from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack.pipeline import BaseStandardPipeline, Pipeline
from haystack.schema import BaseComponent

class TitleQAPipeline(BaseStandardPipeline):
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize a Pipeline for finding documents with a title similar to the query using semantic document search.

        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: Optional[int] = None):
        output = self.pipeline.run(query=query, filters=filters, top_k_retriever=top_k_retriever)
        documents = output["documents"]

        results: Dict = {"query": query, "answers": []}
        for doc in documents:
            cur_answer = {
                "query": doc.meta['name'],
                "answer": doc.text,
                "document_id": doc.id,
                "context": doc.text,
                "score": doc.score,
                "probability": doc.probability,
                "offset_start": 0,
                "offset_end": len(doc.text),
                "meta": doc.meta,
            }

            results["answers"].append(cur_answer)
        return results


class TitleBM25QAPipeline(BaseStandardPipeline):
    def __init__(self, reader: BaseReader, retriever_title: BaseRetriever, retriever_bm25: BaseRetriever, k_title_retriever: int, k_bm25_retriever: int):
        """
        Initialize a Pipeline for Extractive Question Answering. This Pipeline is based on two retrievers and a reader.
        The two retrievers used for this pipeline are :
            - A TitleEmbeddingRetriever
            - An ElasticsearchRetriever
        The output of the two retrievers are concatenated based on the number of k_retriever passed for each retrievers.

        :param reader: Reader instance
        :param retriever: Retriever instance
        :param k_title_retriever: int
        :param k_bm25_retriever: int
        """
        self.k_title_retriever = k_title_retriever
        self.k_bm25_retriever = k_bm25_retriever
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever_bm25, name="Retriever_bm25", inputs=["Query"])
        self.pipeline.add_node(component=retriever_title, name="Retriever_title", inputs=["Query"])
        self.pipeline.add_node(component=JoinDocuments(ks_retriever=[k_bm25_retriever, k_title_retriever]), name="JoinResults",
                               inputs=["Retriever_bm25", "Retriever_title"])
        self.pipeline.add_node(component=reader, name="QAReader", inputs=["JoinResults"])

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: int = 10, top_k_reader: int = 10):
        assert top_k_retriever <= max(self.k_title_retriever, self.k_bm25_retriever), "Be carefull, the pipeline was run with top_k_retriever that is greater than the k_retriever declared at instanciation"
        output = self.pipeline.run(
            query=query, filters=filters, top_k_retriever=top_k_retriever, top_k_reader=top_k_reader
        )
        return output


class JoinDocuments(BaseComponent):
    """
    A node to join documents outputted by multiple retriever nodes.

    The node allows multiple join modes:
    * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    """

    outgoing_edges = 1

    def __init__(
        self, ks_retriever: List[int] = None
    ):
        """
        :param ks_retriever: A node-wise list(length of list must be equal to the number of input nodes) of k_retriever kept for
                        the concatenation of the retrievers in the nodes.
        """
        self.ks_retriever = ks_retriever

    def run(self, **kwargs):
        inputs = kwargs["inputs"]

        #unique mode using ks_retriever
        document_map = {}
        if self.ks_retriever:
            ks_retriever = self.ks_retriever
        else:
            ks_retriever = [len(inputs[0]['documents']) for i in range(len(inputs))]
        for input_from_node, k_retriever in zip(inputs, ks_retriever):
            for i, doc in enumerate(input_from_node["documents"]):
                if i == k_retriever:
                    break
                document_map[doc.id] = doc
        documents = document_map.values()
        output = {"query": inputs[0]["query"], "documents": documents, "labels": inputs[0].get("labels", None)}
        return output, "output_1"