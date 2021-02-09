import pytest
from pathlib import Path

from haystack.document_store.base import BaseDocumentStore
from haystack.pipeline import Pipeline

from src.evaluation.utils.utils_eval import eval_retriever_reader

@pytest.mark.elasticsearch
def test_eval_elastic_retriever_reader(document_store: BaseDocumentStore, retriever_bm25, retriever_emb, reader):
    doc_index = "document"
    label_index = "label"

    p = Pipeline()
    p.add_node(component=retriever_bm25, name="Retriever", inputs=["Query"])
    p.add_node(component=reader, name='Reader', inputs=['Retriever'])

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(filename=Path("./test/samples/squad/tiny.json").as_posix(), doc_index=doc_index,
                                 label_index=label_index)

    assert document_store.get_document_count(index=doc_index) == 3  # number of contexts
    assert document_store.get_label_count(index=label_index) == 20  # number of answers

    # eval retriever
    k_retriever = 3
    retriever_eval_results = eval_retriever_reader(document_store=document_store, pipeline=p,
                                                   top_k_retriever=k_retriever, label_index=label_index)

    assert retriever_eval_results["correct_readings_top1"] == 12
    assert retriever_eval_results["correct_readings_topk"] == 15
    assert retriever_eval_results["correct_readings_top1_has_answer"] == 12
    assert retriever_eval_results["correct_readings_topk_has_answer"] == 15
    assert retriever_eval_results["exact_matches_top1"] == 3
    assert retriever_eval_results["exact_matches_topk"] == 8
    assert retriever_eval_results['reader_topk_accuracy'] == 0.9375 #15/16
    assert retriever_eval_results['reader_topk_accuracy_has_answer'] == 1.0 #15/15

    # clean up
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)