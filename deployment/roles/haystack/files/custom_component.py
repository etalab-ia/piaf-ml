from typing import Dict, List, Optional

from haystack.document_store import ElasticsearchDocumentStore
from haystack.document_store.base import BaseDocumentStore
from haystack.retriever import ElasticsearchRetriever
from haystack.schema import BaseComponent
from haystack.retriever.dense import EmbeddingRetriever
import numpy as np
from haystack import Document


class LabelElasticsearchRetriever(ElasticsearchRetriever):
    """
     A node to search if a question is a perfect match.
     We query the label_elasticsearch index, looking for a question matching the user question.
     If found, we retrieve the first document associated
     If not, we lquery the document_elasticsearch as before
     """

    def __init__(self, document_store: ElasticsearchDocumentStore, top_k: int = 10, custom_query: str = None,
                 weight_when_document_found: int = 99):
        super().__init__(document_store, top_k, custom_query)
        self.weight_when_document_found = weight_when_document_found

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        documents = self.document_store.query(query, None, 1, None, "label_elasticsearch")
        if len(documents) > 0:
            first_doc = documents[0].to_dict()
            documents = self.document_store.get_documents_by_id([first_doc["meta"]["document_id"]],
                                                                "document_elasticsearch")
            doc = documents[0].to_dict()
            doc["meta"]["weight"] = self.weight_when_document_found
            doc = Document.from_dict(doc, field_map={})
            return [doc]
        return []


class TitleEmbeddingRetriever(EmbeddingRetriever):
    def __init__(
            self,
            document_store: BaseDocumentStore,
            embedding_model: str,
            model_version: Optional[str] = None,
            use_gpu: bool = True,
            model_format: str = "farm",
            pooling_strategy: str = "reduce_mean",
            emb_extraction_layer: int = -1,
            top_k: int = 10,
            weight_when_document_found: int = 10
    ):
        super().__init__(document_store, embedding_model, model_version, use_gpu, model_format, pooling_strategy,
                         emb_extraction_layer, top_k)
        self.weight_when_document_found = weight_when_document_found

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        documents = super().retrieve(query, filters, top_k, index)

        for doc in documents:
            doc.meta["weight"] = self.weight_when_document_found

        return documents

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
                doc.meta["answerified"] = True
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

    def __init__(self, threshold_score: float = 0.8, top_k: float = 5, max_reader_answer: int = 1):
        """
        :param threshold_score: The threshold that will be used for keeping or not the answer from the readers
        """
        self.threshold_score = threshold_score
        self.top_k = top_k
        self.max_reader_answer = max_reader_answer

    def run(self, **kwargs):
        inputs = kwargs["inputs"]

        results: Dict = {"query": inputs[0]["query"], "answers": []}

        count_answers = 0
        count_reader = 0
        for input_from_node in inputs:
            for answer in input_from_node['answers']:
                if count_answers == self.top_k:
                    break
                elif answer["score"] is None:  # The answer came from Transformers Reader
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


class MergeOverlappingAnswers(BaseComponent):
    """
    A node that merges two answers when they overlap to avoid having multiple
    answers that are almost identical, like "fichier informatique" and "un petit
    fichier informatique" as responses to the query "qu'est-ce qu'un cookie ?".
    The resulting answer contains both answers merged together.

    Two answers can merge only if they come from the same contexts, i.e. if the
    contexts themselves merge.
    """

    outgoing_edges = 1

    def __init__(self, minimum_overlap_contexts=0.75, minimum_overlap_answers=0.25):
        self.minimum_overlap_contexts = minimum_overlap_contexts
        self.minimum_overlap_answers = minimum_overlap_answers

    def run(self, **kwargs):
        answers = kwargs["answers"]
        merged_answers = []

        for ans in answers:

            if ans["context"] == None or ans["answer"] == None: continue

            # Merge ans with the first possible answer in merged_answers
            is_merged = False
            i = 0
            while not is_merged and i < len(merged_answers):
                mans = merged_answers[i]

                # Try to merge the two contexts
                (new_merged_ctxt, _) = merge_strings(
                    mans["context"],
                    ans["context"],
                    self.minimum_overlap_contexts)

                # If context merge was successful
                if new_merged_ctxt != "":

                    # Try to merge the answers
                    (new_merged_ans, ans_offset) = merge_strings(
                        mans["answer"],
                        ans["answer"],
                        self.minimum_overlap_answers)

                    # If answer merge was successful
                    if new_merged_ans != "":

                        if ans_offset >= 0:
                            offset_start = mans["offset_start"]
                        else:
                            offset_start = mans["offset_start"] + ans_offset

                        offset_end = offset_start + len(new_merged_ans)

                        merged_answers[i] = {
                            'answer': new_merged_ans,
                            'context': new_merged_ctxt,
                            'document_id': mans["document_id"],
                            'meta': mans["meta"],
                            'offset_end': offset_end,
                            'offset_start': offset_start,
                            'probability': max(mans["probability"],
                                               ans["probability"]),
                            'score': None}

                        is_merged = True

                i += 1

            # If ans wasn't merged with anything, add it as is to merged_answers
            if not is_merged:
                merged_answers.append(ans)

        output = kwargs.copy()
        output["answers"] = merged_answers

        return output, "output_1"


class RankAnswersWithWeigth(BaseComponent):
    """
    A node that ranks the answers from the reader considering the weight they have
    """
    outgoing_edges = 1

    def run(self, **kwargs):
        answers = kwargs["answers"]
        answers_dict_format = [a if isinstance(a, dict) else a.to_dict() for a in answers]
        sort_by_weigth(answers_dict_format)
        output = kwargs.copy()
        output["answers"] = answers_dict_format
        return output, "output_1"


def merge_strings(str1, str2, minimum_overlap):
    """
    Returns a tuple `(m, i)` where `m` is a string that is the combination of str1
    and str2 if they overlap, otherwise returns an empty string. The two
    strings are considered to overlap if they are non-empty and one of the
    following conditions apply:
    - one is a substring of the other and it's size >= `minimum_overlap *
      min(len(str1), len(str2))`,
    - the end of one is equal to the beginning of the other and the size of the
      common substring is >= `minimum_overlap * min(len(str1), len(str2))`

    The integer `i` is the offset of str2 relative to str1.

    For example:

    merge_strings("a black", "black coffee", 0.1) == ("a black coffee", 2)
    merge_strings("black coffee", "a black", 0.1) == ("a black coffee", -2)
    merge_strings("a black coffee", "black", 0.1) == ("a black coffee", 2)
    merge_strings("a black coffee", " with milk", 0.1) == ("", 0)
    merge_strings("a black coffee", " with milk", 0) == ("a black coffee with milk", 14)
    merge_strings("a black coffee", "", 0) == ("", 0)
    merge_strings("a coffee is my first thing in the morning", "morning or evening", 0.25) == ("", 0)
    merge_strings("a coffee is my first thing in the morning", "in the morning", 0.25) == ("a coffee is my first thing in the morning", 27)
    """

    if str1 == "" or str2 == "": return ("", 0)

    # Brute force algorithm for a start. Probably inefficient for large
    # sequences.
    # Outline: Start by comparing the end of str1 with the beginning of str2.
    # Shift each sequence towards the other one character at a time. Keep the
    # positions of the longest matching string in both sequences.

    # Best match
    best_i = len(str1)
    best_s = 0

    minimum_overlap_chars = int(minimum_overlap * min(len(str1), len(str2)))

    # i is the number of characters by which to shift the beginning of str2 to
    # the right of the beginning of str1.
    for i in range(len(str1) - 1, -len(str2), -1):

        if i >= 0:
            # Current size of compared substrings
            s = min(len(str1) - i, len(str2))
            # Positions of compared substrings in str1 and str2
            start1 = i
            start2 = 0
        else:  # i < 0
            s = min(len(str2) + i, len(str1))
            start1 = 0
            start2 = -i

        if s >= minimum_overlap_chars \
                and s > best_s \
                and str1[start1: start1 + s] == str2[start2: start2 + s]:
            best_i = i
            best_s = s

    if best_s >= minimum_overlap_chars:
        if best_i >= 0:
            return (str1 + str2[best_s:], best_i)
        else:
            return (str2 + str1[best_s:], best_i)
    else:
        return ("", 0)


def get_weight(doc):
    return doc["meta"].get('weight') or 0


def sort_by_weigth(documents: List[dict]) -> None:
    """
    This function takes a list of "to_dict" documents as input and sort them
    based on a property "weight" located in the "meta" property of the doc.
    Ex: 
    documents = [
        {'Name': 'Alan Turing', 'meta': { 'weight': 10000}},
        {'Name': 'Sharon Lin', 'meta': { 'weight': 8000}},
        {'Name': 'John Hopkins', 'meta': {},
        {'Name': 'Mikhail Tal', 'meta': { 'weight': 15000}}
    ]
    returns after applying sort_by_weigth(documents)
    documents = [
        {'Name': 'Mikhail Tal', 'meta': { 'weight': 15000}},
        {'Name': 'Alan Turing', 'meta': { 'weight': 10000}},
        {'Name': 'Sharon Lin', 'meta': { 'weight': 8000}},
        {'Name': 'John Hopkins', 'meta':{} }
    ]
    """
    documents.sort(key=get_weight, reverse=True)
