
from typing import Optional, Dict

from haystack.retriever.base import BaseRetriever
from haystack.pipeline import BaseStandardPipeline, Pipeline

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