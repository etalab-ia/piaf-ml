from typing import Dict, List
from haystack.schema import BaseComponent
from haystack.retriever.dense import EmbeddingRetriever
import numpy as np
from haystack import Document


class TitleEmbeddingRetriever(EmbeddingRetriever):
    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings of the titles for a list of passages. For this Retriever type: The same as calling .embed()
        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        texts = [d.meta["name"] for d in docs]

        return self.embedding_encoder.embed(texts)


class JoinDocumentsCustom(BaseComponent):
    """
    A node to join documents outputted by multiple retriever nodes.
    The node allows multiple join modes:
    * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    """

    outgoing_edges = 1

    def __init__(self, ks_retriever: List[int] = None):
        """
        :param ks_retriever: A node-wise list(length of list must be equal to the number of input nodes) of k_retriever kept for
                        the concatenation of the retrievers in the nodes. If set to None, the number of documents retrieved will be used
        """
        self.ks_retriever = ks_retriever

    def run(self, **kwargs):
        inputs = kwargs["inputs"]

        document_map = {}
        if self.ks_retriever:
            ks_retriever = self.ks_retriever
        else:
            ks_retriever = [len(inputs[0]["documents"]) for i in range(len(inputs))]
        for input_from_node, k_retriever in zip(inputs, ks_retriever):
            for i, doc in enumerate(input_from_node["documents"]):
                if i == k_retriever:
                    break
                document_map[doc.id] = doc
        documents = document_map.values()
        output = {
            "query": inputs[0]["query"],
            "documents": documents,
            "labels": inputs[0].get("labels", None),
        }
        return output, "output_1"

class AnswerifyDocuments(BaseComponent):
    """
    This component is used to transform the documents retrieved in a shape that can be used like a Reader answer.
    """
    outgoing_edges = 1

    def run(self, **kwargs):
        query = kwargs["query"]
        documents = kwargs["documents"]
        results: Dict = {"query": query, "answers": []}
        if documents:

            for doc in documents:
                cur_answer = {
                    "query": doc.meta["name"],
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

        return results, "output_1"


class JoinAnswers(BaseComponent):
    """
        A node to join documents outputted by multiple reader nodes.

        The node allows multiple join modes:

        * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
        * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
          `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
        """

    outgoing_edges = 1

    def __init__(self, threshold_score: float = 0.8):
        """
        :param threshold_score: The threshold that will be used for keeping or not the answer from the readers
        """
        self.threshold_score = threshold_score
        self.max_reader_answer = 1 # Only one answer given from reader

    def run(self, **kwargs):
        inputs = kwargs["inputs"]
        answers_inputs = []
        for input in inputs:
            if 'pipeline_type' in input.keys(): #detects the 'Query' input
                top_k_reader = inputs[0]["top_k_reader"]
            else:
                answers_inputs.append(input) # only keep the inputs from the readers

        results: Dict = {"query": inputs[0]["query"], "answers": []}

        count_answers = 0
        count_reader = 0
        for input_from_node in answers_inputs:
            for answer in input_from_node['answers']:
                if count_answers == top_k_reader:
                    break
                elif answer["score"] is None: #The answer came from Transformers Reader
                    if answer["probability"] > self.threshold_score and count_reader < self.max_reader_answer:
                        if answer["answer"] is not None:
                            results["answers"].append(answer)
                            count_answers += 1
                            count_reader += 1
                    continue
                else:
                    if answer["answer"] is not None:
                        results["answers"].append(answer)
                        count_answers += 1

        return results, "output_1"