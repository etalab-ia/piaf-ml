from typing import List
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




class MergeOverlappingAnswers():
    """
    A node that merges two answers when they overlap to avoid having multiple
    answers that are almost identical, like "fichier informatique" and "un petit
    fichier informatique" as responses to the query "qu'est-ce qu'un cookie ?".
    The resulting answer contains both answers merged together.

    Two answers can merge only if they come from the same contexts, i.e. if the
    contexts themselves merge.
    """

    outgoing_edges=1

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
                new_merged_ctxt = merge_strings(mans["context"], ans["context"])
                if new_merged_ctxt != "":
                    new_merged_ans = merge_strings(mans["answer"], ans["answer"])
                    if new_merged_ans != "":
                        offset_start = new_merged_ctxt.find(new_merged_ans)
                        merged_answers[i] = {
                              'answer': new_merged_ans,
                              'context': new_merged_ctxt,
                              'document_id': mans["document_id"],
                              'meta': mans["meta"],
                              'offset_end': offset_start + len(new_merged_ans),
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




def merge_strings(str1, str2):
    """
    Returns (m, s) where m is a string that is the combination of str1 and str2
    if they overlap and s is the overlap size. If they don't overlap, m an
    empty string. For example:

    merge_strings("a black", "black coffee") == "a black coffee"
    merge_strings("a black coffee", "black") == "a black coffee"
    merge_strings("a black coffee", "") == ""
    """

    if str1 == "" or str2 == "": return ""

    # Brute force algorithm for a start. Probably inefficient for large 
    # sequences.
    # Outline: Start by comparing the end of str1 with the beginning of str2. 
    # Shift each sequence towards the other one character at a time. Keep the
    # positions of the longest matching string in both sequences.

    # Best match
    best_i = len(str1)
    best_s = 0

    # i is the number of characters by which to shift the beginning of str2 to
    # the right of the beginning of str1.
    for i in range(len(str1) - 1, -len(str2), -1):

        if i >= 0:
            # Current size of compared substrings
            s = min(len(str1) - i, len(str2))
            # Positions of compared substrings in str1 and str2
            start1 = i
            start2 = 0
        else: # i < 0
            s = min(len(str2) + i, len(str1))
            start1 = 0
            start2 = -i

        if s > best_s \
                and str1[start1 : start1 + s] == str2[start2 : start2 + s]:
            best_i = i
            best_s = s

    if best_i >= 0:
        return str1 + str2[best_s:]
    else:
        return str2 + str1[best_s:]
