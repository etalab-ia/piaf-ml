import pytest
from pathlib import Path

from haystack.document_store.base import BaseDocumentStore
from haystack.pipeline import Pipeline

from src.evaluation.utils.utils_eval import eval_faq_pipeline
from src.evaluation.utils.FAQPipeline import FAQPipeline

@pytest.mark.elasticsearch
def test_eval_elastic_retriever_reader(document_store: BaseDocumentStore, retriever_faq):
    doc_index = "document"
    label_index = "label"

    p = FAQPipeline(retriever_faq)

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(filename=Path("./test/samples/squad/faq.json").as_posix(), doc_index=doc_index,
                                 label_index=label_index)

    document_store.update_embeddings(retriever_faq, index=doc_index)

    assert document_store.get_document_count(index=doc_index) == 9  # number of contexts
    assert document_store.get_label_count(index=label_index) == 9  # number of answers

    # eval retriever
    k_retriever = 3
    retriever_eval_results = eval_faq_pipeline(document_store=document_store, pipeline=p,
                                                   k_retriever=k_retriever,
                                                   label_index=label_index)

    assert retriever_eval_results["correct_readings_top1"] == 7
    assert retriever_eval_results["correct_readings_topk"] == 9
    assert retriever_eval_results['reader_topk_accuracy'] == 1.
    assert retriever_eval_results['reader_top1_accuracy'] == 7/9


    # clean up
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)