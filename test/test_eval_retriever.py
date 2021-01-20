import pytest
from pathlib import Path

from haystack.document_store.base import BaseDocumentStore
from haystack.pipeline import Pipeline

from src.evaluation.utils.utils_eval import eval_retriever


@pytest.mark.elasticsearch
def test_add_eval_data(document_store):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename=Path("./test/samples/squad/small.json").as_posix(),
                                 doc_index="test_eval_document", label_index="test_feedback")

    assert document_store.get_document_count(index="test_eval_document") == 11
    assert document_store.get_label_count(index="test_feedback") == 65

    # test documents
    docs = document_store.get_all_documents(index="test_eval_document")
    assert docs[0].text[:10] == "Les dépens"
    assert docs[0].meta["name"] == "Sport"
    assert len(docs[0].meta.keys()) == 2

    # test labels
    labels = document_store.get_all_labels(index="test_feedback")
    assert labels[0].answer == "100 000"
    assert labels[0].no_answer is False
    assert labels[0].is_correct_answer is True
    assert labels[0].is_correct_document is True
    assert labels[0].question == 'Combien de personnes travaillent au ministère des sports'
    assert labels[0].origin == "gold_label"
    assert labels[0].offset_start_in_doc == 472

    # check combination
    assert labels[0].document_id == docs[0].id
    start = labels[0].offset_start_in_doc
    end = start + len(labels[0].answer)
    assert docs[0].text[start:end] == "100 000"

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")


@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_type,score_expected", [("bm25", 14 / 15), ("sbert", 13 / 15)])
def test_eval_elastic_retriever(document_store: BaseDocumentStore, retriever_bm25, retriever_emb, retriever_type,
                                score_expected):
    doc_index = "document"
    label_index = "label"

    p = Pipeline()
    if retriever_type == 'bm25':
        retriever = retriever_bm25
        p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    elif retriever_type == "sbert":
        retriever = retriever_emb
        p.add_node(component=retriever, name="SBertRetriever", inputs=["Query"])
    else:
        raise Exception(f"You chose {retriever_type}. Choose one from bm25, sbert, or dpr")

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(filename=Path("./test/samples/squad/tiny.json").as_posix(), doc_index=doc_index,
                                 label_index=label_index)
    document_store.update_embeddings(retriever_emb, index=doc_index)
    assert document_store.get_document_count(index=doc_index) == 3  # number of contexts
    assert document_store.get_label_count(index=label_index) == 18  # number of answers

    # eval retriever
    retriever_eval_results = eval_retriever(document_store=document_store, pipeline=p, top_k=3, label_index=label_index,
                                            doc_index=doc_index)

    assert retriever_eval_results["recall"] == 1.0
    assert retriever_eval_results["mrr"] == score_expected

    # clean up
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
