import pytest
from pathlib import Path

from haystack.document_store.base import BaseDocumentStore
from haystack.finder import Finder


@pytest.mark.elasticsearch
def test_add_eval_data(document_store):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename=Path("./test/samples/squad/small.json"), doc_index="test_eval_document", label_index="test_feedback")

    assert document_store.get_document_count(index="test_eval_document") == 10
    assert document_store.get_label_count(index="test_feedback") == 64

    # test documents
    docs = document_store.get_all_documents(index="test_eval_document")
    assert docs[0].text[:10] == "Les dépens"
    assert docs[0].meta["name"] == "Sport"
    assert len(docs[0].meta.keys()) == 2

    # test labels
    labels = document_store.get_all_labels(index="test_feedback")
    assert labels[0].answer == "100 000"
    assert labels[0].no_answer == False
    assert labels[0].is_correct_answer == True
    assert labels[0].is_correct_document == True
    assert labels[0].question == 'Combien de personnes travaillent au ministère des sports'
    assert labels[0].origin == "gold_label"
    assert labels[0].offset_start_in_doc == 472

    # check combination
    assert labels[0].document_id == docs[0].id
    start = labels[0].offset_start_in_doc
    end = start+len(labels[0].answer)
    assert docs[0].text[start:end] == "100 000"

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")

"""
@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("open_domain", [True, False])
@pytest.mark.parametrize("retriever", ["elasticsearch"], indirect=True)
def test_eval_elastic_retriever(document_store: BaseDocumentStore, open_domain, retriever):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename="samples/squad/tiny.json", doc_index="test_eval_document", label_index="test_feedback")
    assert document_store.get_document_count(index="test_eval_document") == 2

    # eval retriever
    results = retriever.eval(top_k=1, label_index="test_feedback", doc_index="test_eval_document", open_domain=open_domain)
    assert results["recall"] == 1.0
    assert results["mrr"] == 1.0
    if not open_domain:
        assert results["map"] == 1.0

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")"""

