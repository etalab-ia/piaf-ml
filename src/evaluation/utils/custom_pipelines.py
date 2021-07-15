from copy import deepcopy
from typing import Dict, List, Optional

from haystack.pipeline import BaseStandardPipeline, Pipeline
from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack.schema import BaseComponent

from deployment.roles.haystack.files.custom_component import \
    MergeOverlappingAnswers, JoinDocumentsCustom


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
            component=JoinDocumentsCustom(ks_retriever=[k_bm25_retriever, k_title_retriever]),
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
        self.pipeline.add_node(component=JoinDocumentsCustom(ks_retriever=[k_bm25_retriever, k_title_retriever]), name="JoinResults",
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
