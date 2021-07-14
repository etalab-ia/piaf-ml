from copy import deepcopy
from typing import Dict, List, Optional

from haystack.pipeline import BaseStandardPipeline, Pipeline
from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack.schema import BaseComponent


class TitleQAPipeline(BaseStandardPipeline):
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize a Pipeline for finding documents with a title similar to the query using semantic document search.

        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k_retriever: Optional[int] = None,
    ):
        output = self.pipeline.run(
            query=query, filters=filters, top_k_retriever=top_k_retriever
        )
        documents = output["documents"]

        results: Dict = {"query": query, "answers": []}
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
        return results


class TitleBM25QAPipeline(BaseStandardPipeline):
    def __init__(
        self,
        reader: BaseReader,
        retriever_title: BaseRetriever,
        retriever_bm25: BaseRetriever,
        k_title_retriever: int,
        k_bm25_retriever: int,
    ):
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
        self.pipeline.add_node(
            component=retriever_bm25, name="Retriever_bm25", inputs=["Query"]
        )
        self.pipeline.add_node(
            component=retriever_title, name="Retriever_title", inputs=["Query"]
        )
        self.pipeline.add_node(
            component=JoinDocuments(ks_retriever=[k_bm25_retriever, k_title_retriever]),
            name="JoinResults",
            inputs=["Retriever_bm25", "Retriever_title"],
        )
        self.pipeline.add_node(
            component=reader, name="QAReader", inputs=["JoinResults"]
        )

    def run(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k_retriever: int = 10,
        top_k_reader: int = 10,
    ):
        assert top_k_retriever <= max(
            self.k_title_retriever, self.k_bm25_retriever
        ), "Be carefull, the pipeline was run with top_k_retriever that is greater than the k_retriever declared at instanciation"
        output = self.pipeline.run(
            query=query,
            filters=filters,
            top_k_retriever=top_k_retriever,
            top_k_reader=top_k_reader,
        )
        return output


class RetrieverReaderEvaluationPipeline(BaseStandardPipeline):
    def __init__(self, reader: BaseReader, retriever: BaseRetriever , eval_retriever , eval_reader):
        """
        Initialize an Evaluation Pipeline for Extractive Question Answering. This Pipeline is based on retriever reader architecture.
        it includes two evaluation nodes :
            - An EvalRetriever node after Retriever
            - An EvalReader node after RetrReader

        :param reader: Reader instance
        :param retriever: Retriever instance
        :param eval_retriever : EvalRetriever instance
        :param eval_reader : EvalReader instance
        """

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Retriever"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["EvalRetriever"])
        self.pipeline.add_node(component=MergeOverlappingAnswers(), name="MergeOverlappingAnswers", inputs=["Reader"])
        self.pipeline.add_node(component=eval_reader, name="EvalReader", inputs=["Reader"])


    def run(self, query, top_k_retriever, top_k_reader, labels):
        """
        run function definition of the customized RetrieverReaderEvaluationPipeline 

        :param query: string  (question or query)
        :param top_k_retriever: int
        :param top_k_reader : int
        :param labels : Dict of multilabel (has the form {{'retriever':multilabel},'reader':multilabel}})
        """

        output = self.pipeline.run(
            query=query,
            top_k_retriever=top_k_retriever,
            labels=labels,
            top_k_reader=top_k_reader,
        )

        return output


class TitleBM25QAEvaluationPipeline(BaseStandardPipeline):
    def __init__(self, 
                reader: BaseReader, 
                retriever_title: BaseRetriever, 
                retriever_bm25: BaseRetriever, 
                k_title_retriever: int, 
                k_bm25_retriever: int,
                eval_retriever ,
                eval_reader
                ):

        """
        Initialize an Evaluation Pipeline for Extractive Question Answering. This Pipeline is based on on two retrievers and a reader.
        The two retrievers used for this pipeline are :
            - A TitleEmbeddingRetriever
            - An ElasticsearchRetriever
            
        it includes two evaluation nodes :
            - An EvalRetriever node after Retriever
            - An EvalReader node after RetrReader

        :param reader: Reader instance
        :param retriever: Retriever instance
        :param eval_retriever : EvalRetriever instance
        :param eval_reader : EvalReader instance
        """
       
        self.k_title_retriever = k_title_retriever
        self.k_bm25_retriever = k_bm25_retriever
        self.pipeline = Pipeline()

        self.pipeline.add_node(component=retriever_bm25, name="Retriever_bm25", inputs=["Query"])
        self.pipeline.add_node(component=retriever_title, name="Retriever_title", inputs=["Query"])
        self.pipeline.add_node(component=JoinDocuments(ks_retriever=[k_bm25_retriever, k_title_retriever]), name="JoinResults",
                               inputs=["Retriever_bm25", "Retriever_title"])

        self.pipeline.add_node(component=eval_retriever, name="EvalRetriever", inputs=["JoinResults"])
        self.pipeline.add_node(component=reader, name="QAReader", inputs=["EvalRetriever"])

        self.pipeline.add_node(component=eval_reader, name="EvalReader", inputs=["QAReader"])

    def run(self, query, top_k_retriever, top_k_reader, labels):

        assert top_k_retriever <= max(self.k_title_retriever, self.k_bm25_retriever), "Be carefull, the pipeline was run with top_k_retriever that is greater than the k_retriever declared at instanciation"
        
        output = self.pipeline.run(
            query=query,
            top_k_retriever=top_k_retriever,
            labels=labels,
            top_k_reader=top_k_reader,
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

    def __init__(self, ks_retriever: List[int] = None):
        """
        :param ks_retriever: A node-wise list(length of list must be equal to the number of input nodes) of k_retriever
        kept for the concatenation of the retrievers in the nodes. If set to None, the number of documents retrieved
        will be used
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
