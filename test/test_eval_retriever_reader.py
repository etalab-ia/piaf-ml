import pytest
from pathlib import Path

from haystack.document_store.base import BaseDocumentStore
from haystack.pipeline import Pipeline

from src.evaluation.utils.utils_eval import full_eval_retriever_reader 

@pytest.mark.elasticsearch
@pytest.mark.parametrize("k_reader_total", [10, 1])
def test_eval_elastic_retriever_reader(document_store: BaseDocumentStore, retriever_bm25, reader, k_reader_total, Eval_Retriever ,Eval_Reader):
    doc_index = "document"
    label_index = "label"

    p = Pipeline()
    p.add_node(component=retriever_bm25, name="Retriever", inputs=["Query"])
    p.add_node(component=Eval_Retriever, name='EvalRetriever', inputs=['Retriever'] )
    p.add_node(component=reader, name='Reader', inputs=['EvalRetriever'])
    p.add_node(component=Eval_Reader, name='EvalReader', inputs=['Reader'])

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(filename=Path("./test/samples/squad/tiny.json").as_posix(), doc_index=doc_index,
                                 label_index=label_index)

    assert document_store.get_document_count(index=doc_index) == 3  # number of contexts
    assert document_store.get_label_count(index=label_index) == 20  # number of answers

    # eval retriever
    k_retriever = 3
    full_eval_retriever_reader(document_store=document_store, pipeline=p,
                               k_retriever=k_retriever, k_reader_total=k_reader_total,
                               label_index=label_index)

    retriever_eval_results =  Eval_Retriever.get_metrics()
    retriever_eval_results.update(Eval_Reader.get_metrics())


    """For 16 queries: 
        13 Queries : documents retrieved at position 1
        2  Queries : documents retrieved at position 2
        1  Queries : No documents Retrieved """

    assert retriever_eval_results["recall"] == 0.9375 #15/16
    assert retriever_eval_results["map"] == 0.875  # 14/16
    assert retriever_eval_results["mrr"] == 0.875  # 14/16 


    if k_reader_total == 10:
        assert retriever_eval_results["correct_readings_top1"] == 12
        assert retriever_eval_results["correct_readings_topk"] == 15
        assert retriever_eval_results["correct_readings_top1_has_answer"] == 12
        assert retriever_eval_results["correct_readings_topk_has_answer"] == 15
        assert retriever_eval_results["exact_matches_top1"] == 3
        assert retriever_eval_results["exact_matches_topk"] == 8
        assert retriever_eval_results['reader_topk_accuracy'] == 0.9375 #15/16
        assert retriever_eval_results['reader_topk_accuracy_has_answer'] == 1.0 #15/15
        assert retriever_eval_results['reader_topk_accuracy_has_answer'] == 1.0 #15/15
    elif k_reader_total == 1:
        assert retriever_eval_results["correct_readings_top1"] == retriever_eval_results["correct_readings_topk"]
        assert retriever_eval_results["correct_readings_top1_has_answer"] == retriever_eval_results["correct_readings_topk_has_answer"]
        assert retriever_eval_results["exact_matches_top1"] == retriever_eval_results["exact_matches_topk"]

    # clean up
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)