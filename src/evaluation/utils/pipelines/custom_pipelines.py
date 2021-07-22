"""
This module contains functions that return haystack pipelines or classes that
are subclasses of BaseStandardPipeline.
"""

from copy import deepcopy
from typing import Dict, List, Optional

from haystack.pipeline import BaseStandardPipeline, Pipeline
from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack.schema import BaseComponent

from deployment.roles.haystack.files.custom_component import \
    MergeOverlappingAnswers, JoinDocumentsCustom, JoinAnswers, \
    AnswerifyDocuments, StripLeadingSpace

import src.evaluation.utils.pipelines.components.document_stores as document_stores
import src.evaluation.utils.pipelines.components.evals as evals
import src.evaluation.utils.pipelines.components.readers as readers
import src.evaluation.utils.pipelines.components.retrievers as retrievers

# TODO: Obsolete?
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


# TODO: Obsolete?
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

        :param reader: Reader instance :param retriever: Retriever instance :param k_title_retriever: int :param
        k_bm25_retriever: int
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




def retriever_reader(reader, retriever, eval_retriever, eval_reader):
    """
    Returns an Evaluation Pipeline for Extractive Question Answering. This Pipeline is based on retriever reader architecture.
    it includes two evaluation nodes :
        - An EvalRetriever node after Retriever
        - An EvalReader node after RetrReader

    :param reader: Reader instance
    :param retriever: Retriever instance
    :param eval_retriever: EvalRetriever instance or None
    :param eval_reader: EvalReader instance or None
    """

    pipeline = Pipeline()
    pipeline.add_node(
            component=retriever,
            name="Retriever",
            inputs=["Query"])

    if eval_retriever:
        pipeline.add_node(
                component=eval_retriever,
                name="EvalRetriever",
                inputs=["Retriever"])
        pipeline.add_node(
                component=reader,
                name="Reader",
                inputs=["EvalRetriever"])
    else:
        pipeline.add_node(
                component=reader,
                name="Reader",
                inputs=["Retriever"])

    pipeline.add_node(
            component=MergeOverlappingAnswers(),
            name="MergeOverlappingAnswers",
            inputs=["Reader"])
    pipeline.add_node(
            component=StripLeadingSpace(),
            name='StripLeadingSpace',
            inputs=['MergeOverlappingAnswers'])

    if eval_reader:
        pipeline.add_node(
                component=eval_reader,
                name="EvalReader",
                inputs=["StripLeadingSpace"])

    return pipeline



def retriever_bm25(
        elasticsearch_hostname,
        elasticsearch_port,
        title_boosting_factor):

    document_store = document_stores.elasticsearch(
            elasticsearch_hostname, elasticsearch_port,
            similarity = "cosine", embedding_dim = 768,
            title_boosting_factor = title_boosting_factor)

    retriever = retrievers.bm25(document_store=document_store)

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])

    return p




def retriever_sbert(
        elasticsearch_hostname,
        elasticsearch_port,
        title_boosting_factor,
        retriever_model_version,
        gpu_available):

    document_store = document_stores.elasticsearch(
        elasticsearch_hostname, elasticsearch_port,
        similarity = "cosine", embedding_dim = 768,
        title_boosting_factor = title_boosting_factor)

    retriever = retrievers.sbert(document_store, retriever_model_version,
            gpu_available)

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])

    return p




def retriever_google(
            elasticsearch_hostname,
            elasticsearch_port,
            title_boosting_factor,
            google_retriever_website):

        document_store = document_stores.elasticsearch(
                elasticsearch_hostname, elasticsearch_port,
                similarity = "cosine", embedding_dim = 768,
                title_boosting_factor = title_boosting_factor)

        retriever = retrievers.google(document_store, google_retriever_website)

        p = Pipeline()
        p.add_node(component=retriever, name="Retriever", inputs=["Query"])

        return p




def retriever_epitca(
            elasticsearch_hostname,
            elasticsearch_port,
            title_boosting_factor):

        document_store = document_stores.elasticsearch(
                elasticsearch_hostname, elasticsearch_port,
                similarity = "cosine", embedding_dim = 768,
                title_boosting_factor = title_boosting_factor)

        retriever = retrievers.epitca(document_store)

        p = Pipeline()
        p.add_node(component=retriever, name="Retriever", inputs=["Query"])

        return p




def retriever_reader_bm25(
        elasticsearch_hostname,
        elasticsearch_port,
        title_boosting_factor,
        reader_model_version,
        gpu_id,
        k_reader_per_candidate):

    document_store = document_stores.elasticsearch(
            elasticsearch_hostname, elasticsearch_port,
            similarity = "cosine", embedding_dim = 768,
            title_boosting_factor = title_boosting_factor)

    retriever = retrievers.bm25(document_store)

    reader = readers.transformers_reader(reader_model_version, gpu_id,
            k_reader_per_candidate)

    eval_retriever = evals.piaf_eval_retriever()
    eval_reader = evals.piaf_eval_reader()

    return retriever_reader(
            reader = reader,
            retriever = retriever,
            eval_retriever = eval_retriever,
            eval_reader = eval_reader)




def retriever_reader_sbert(
        elasticsearch_hostname,
        elasticsearch_port,
        title_boosting_factor,
        retriever_model_version,
        reader_model_version,
        gpu_id,
        k_reader_per_candidate):

    document_store = document_stores.elasticsearch(
            elasticsearch_hostname, elasticsearch_port,
            similarity = "cosine", embedding_dim = 768,
            title_boosting_factor = title_boosting_factor)

    gpu_available = gpu_id >= 0

    retriever = retrievers.sbert(document_store, retriever_model_version,
            gpu_available)

    reader = readers.transformers_reader(reader_model_version, gpu_id, 
            k_reader_per_candidate)

    eval_retriever = evals.piaf_eval_retriever()
    eval_reader = evals.piaf_eval_reader()

    return retriever_reader(
            reader = reader,
            retriever = retriever,
            eval_retriever = eval_retriever,
            eval_reader = eval_reader)




def retriever_reader_dpr(
        elasticsearch_hostname,
        elasticsearch_port,
        title_boosting_factor,
        dpr_model_version,
        reader_model_version,
        gpu_id,
        k_reader_per_candidate):

    document_store = document_stores.elasticsearch(
            elasticsearch_hostname, elasticsearch_port,
            similarity = "dot_product", embedding_dim = 768,
            title_boosting_factor = title_boosting_factor)

    gpu_available = gpu_id >= 0

    retriever = retrievers.dpr(document_store, dpr_model_version,
            gpu_available)

    reader = readers.transformers_reader(reader_model_version, gpu_id,
            k_reader_per_candidate)

    eval_retriever = evals.piaf_eval_retriever()
    eval_reader = evals.piaf_eval_reader()

    return retriever_reader(
            reader = reader,
            retriever = retriever,
            eval_retriever = eval_retriever,
            eval_reader = eval_reader)




def retriever_reader_title_bm25(
        elasticsearch_hostname,
        elasticsearch_port,
        title_boosting_factor,
        retriever_model_version,
        reader_model_version,
        gpu_id,
        k_reader_per_candidate,
        k_title_retriever,
        k_bm25_retriever):
    """
    Returns an Evaluation Pipeline for Extractive Question Answering. This Pipeline is based on on two retrievers and a reader.
    The two retrievers used for this pipeline are :
        - A TitleEmbeddingRetriever
        - An ElasticsearchRetriever

    it includes two evaluation nodes :
        - An EvalRetriever node after Retriever
        - An EvalReader node after RetrReader
    """

    document_store = document_stores.elasticsearch(
            elasticsearch_hostname, elasticsearch_port,
            similarity = "cosine", embedding_dim = 768,
            title_boosting_factor = title_boosting_factor)

    gpu_available = gpu_id >= 0

    retriever_title = retrievers.title(document_store, retriever_model_version,
            gpu_available)

    retriever_bm25 = retrievers.bm25(document_store)

    reader = readers.transformers_reader(reader_model_version, gpu_id, 
            k_reader_per_candidate)

    eval_retriever = evals.piaf_eval_retriever()
    eval_reader = evals.piaf_eval_reader()

    pipeline = Pipeline()
    pipeline.add_node(
            component=retriever_bm25,
            name="Retriever_bm25",
            inputs=["Query"])
    pipeline.add_node(
            component=retriever_title,
            name="Retriever_title",
            inputs=["Query"])
    pipeline.add_node(
            component=JoinDocumentsCustom(ks_retriever=[k_bm25_retriever,
                k_title_retriever]),
            name="JoinResults",
            inputs=["Retriever_bm25", "Retriever_title"])

    if eval_retriever:
        pipeline.add_node(
                component=eval_retriever,
                name="EvalRetriever",
                inputs=["JoinResults"])
        pipeline.add_node(
                component=reader,
                name="Reader",
                inputs=["EvalRetriever"])
    else:
        pipeline.add_node(
                component=reader,
                name="Reader",
                inputs=["JoinResults"])

    pipeline.add_node(
            component=MergeOverlappingAnswers(),
            name="MergeOverlappingAnswers",
            inputs=["Reader"])
    pipeline.add_node(
            component=StripLeadingSpace(),
            name='StripLeadingSpace',
            inputs=['MergeOverlappingAnswers'])

    if eval_reader:
        pipeline.add_node(
                component=eval_reader,
                name="EvalReader",
                inputs=["StripLeadingSpace"])

    return pipeline




def retriever_reader_title(
        elasticsearch_hostname,
        elasticsearch_port,
        title_boosting_factor,
        retriever_model_version,
        reader_model_version,
        gpu_id,
        k_reader_per_candidate):

    document_store = document_stores.elasticsearch(
            elasticsearch_hostname, elasticsearch_port,
            similarity = "cosine", embedding_dim = 768,
            title_boosting_factor = title_boosting_factor)

    gpu_available = gpu_id >= 0

    retriever = retrievers.title(document_store, retriever_model_version,
            gpu_available)

    reader = readers.transformers_reader(reader_model_version, gpu_id,
            k_reader_per_candidate)

    eval_retriever = evals.piaf_eval_retriever()
    eval_reader = evals.piaf_eval_reader()

    return retriever_reader(
            reader = reader,
            retriever = retriever,
            eval_retriever = eval_retriever,
            eval_reader = eval_reader)




def hottest_reader_pipeline(
            elasticsearch_hostname,
            elasticsearch_port,
            title_boosting_factor,
            retriever_model_version,
            reader_model_version,
            gpu_id,
            k_reader_per_candidate,
            k_title_retriever,
            k_bm25_retriever,
            threshold_score):
    """
    Initialize a Pipeline for Extractive Question Answering. This Pipeline is based on two retrievers and a reader.
    The two retrievers used for this pipeline are :

        - A TitleEmbeddingRetriever
        - An ElasticsearchRetriever
    The output of the two retrievers are concatenated based on the number of k_retriever passed for each retrievers.

    :param reader: Reader instance :param retriever: Retriever instance :param k_title_retriever: int :param
    k_bm25_retriever: int
    """

    document_store = document_stores.elasticsearch(
            elasticsearch_hostname = elasticsearch_hostname,
            elasticsearch_port = elasticsearch_port,
            similarity = "cosine",
            embedding_dim = 768,
            title_boosting_factor = title_boosting_factor)

    retriever_title = retrievers.title(
            document_store = document_store,
            retriever_model_version = retriever_model_version,
            gpu_available = gpu_id >= 0)

    retriever_bm25 = retrievers.bm25(document_store)

    reader = readers.transformers_reader(reader_model_version, gpu_id,
            k_reader_per_candidate)

    pipeline = Pipeline()
    pipeline.add_node(
        component=retriever_bm25,
        name="Retriever_bm25",
        inputs=["Query"])
    pipeline.add_node(
        component=retriever_title,
        name="Retriever_title",
        inputs=["Query"])
    pipeline.add_node(
        component=JoinDocumentsCustom(ks_retriever=[k_bm25_retriever,
            k_title_retriever]),
        name="JoinRetrieverResults",
        inputs=["Retriever_bm25", "Retriever_title"])
    pipeline.add_node(
        component=reader,
        name="Reader",
        inputs=["JoinRetrieverResults"])
    pipeline.add_node(
        component=AnswerifyDocuments(),
        name="AnswerFromRetrievers",
        inputs=["JoinRetrieverResults"])
    pipeline.add_node(
        component=JoinAnswers(threshold_score=threshold_score),
        name="JoinResults", 
        inputs=["Query","Reader", "AnswerFromRetrievers"])
    pipeline.add_node(
            component=MergeOverlappingAnswers(),
            name="MergeOverlappingAnswers",
            inputs=["Reader"])
    pipeline.add_node(component=StripLeadingSpace(),
            name='StripLeadingSpace',
            inputs=['MergeOverlappingAnswers'])

    return pipeline


