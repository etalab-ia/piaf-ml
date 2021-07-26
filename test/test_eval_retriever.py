import pytest
from pathlib import Path

from haystack.document_store.base import BaseDocumentStore
from haystack.pipeline import Pipeline

from src.evaluation.utils.elasticsearch_management import delete_indices
from src.evaluation.utils.utils_eval import eval_retriever
from src.data.evaluation_datasets import prepare_fquad_eval


@pytest.mark.elasticsearch
def test_add_eval_data(document_store):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(
        filename=Path("./test/samples/squad/small.json").as_posix(),
        doc_index="test_eval_document",
        label_index="test_feedback",
    )

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
    assert (
        labels[0].question == "Combien de personnes travaillent au ministère des sports"
    )
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
def test_add_eval_data_with_preprocessor(document_store, preprocessor):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(
        filename=Path("./test/samples/squad/small.json").as_posix(),
        doc_index="test_eval_document",
        label_index="test_feedback",
        preprocessor=preprocessor,
    )

    assert document_store.get_document_count(index="test_eval_document") == 38
    assert document_store.get_label_count(index="test_feedback") == 65

    # test documents
    docs = document_store.get_all_documents(index="test_eval_document")
    assert docs[0].text[:10] == "Les dépens"
    assert docs[0].meta["name"] == "Sport"
    assert len(docs[0].meta.keys()) == 4
    assert docs[2].meta["_split_offset"] == 584

    # test labels
    labels = document_store.get_all_labels(index="test_feedback")
    assert labels[1].answer == "20 000"
    assert labels[1].no_answer is False
    assert labels[1].is_correct_answer is True
    assert labels[1].is_correct_document is True
    assert labels[1].question == "Combien d'employeurs"
    assert labels[1].origin == "gold_label"
    assert labels[1].offset_start_in_doc == 13

    # check combination
    assert labels[1].document_id == docs[2].id
    start = labels[1].offset_start_in_doc
    end = start + len(labels[1].answer)
    assert docs[2].text[start:end] == "20 000"

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")


def test_prepare_fquad_eval(document_store):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")

    fquad_eval_file = Path("./output/test/sample/squad/fquad_eval.json")
    fquad_eval_file.parent.mkdir(parents=True, exist_ok=True)

    prepare_fquad_eval.main(
        file_kb_fquad = Path("./test/samples/squad/small.json"),
        file_test_fquad = Path("./test/samples/squad/tiny.json"),
        modified_fquad_path = fquad_eval_file
    )
    document_store.add_eval_data(
        filename = fquad_eval_file.as_posix(),
        doc_index="test_eval_document",
        label_index="test_feedback",
    )

    assert document_store.get_document_count(index="test_eval_document") == 11
    assert document_store.get_label_count(index="test_feedback") == 54

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")


@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_type,recall_expected,mrr_expected",
                         [
                             ("bm25", 15 / 16, 14 / 16),
                             ("sbert", 1., (3 * 1 / 2 + 13) / 16),
                             # ("sbert", 1, (5 * 1 / 2 + 11) / 16)]),
                             ("dpr", 1., (1 * 1 / 2 + 15) / 16)
                         ])
def test_eval_elastic_retriever(document_store: BaseDocumentStore, retriever_bm25, retriever_emb, retriever_dpr,
                                retriever_type, recall_expected, mrr_expected):
    doc_index = "document"
    label_index = "label"

    p = Pipeline()
    if retriever_type == "bm25":
        retriever = retriever_bm25
        p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    elif retriever_type == "sbert":
        retriever = retriever_emb
        p.add_node(component=retriever, name="SBertRetriever", inputs=["Query"])
    elif retriever_type == "dpr":
        retriever = retriever_dpr
        p.add_node(component=retriever, name="DPRRetriever", inputs=["Query"])
    else:
        raise Exception(
            f"You chose {retriever_type}. Choose one from bm25, sbert, or dpr"
        )

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(filename=Path("./test/samples/squad/tiny.json").as_posix(), doc_index=doc_index,
                                 label_index=label_index)
    if retriever_type == "sbert":
        document_store.update_embeddings(retriever_emb, index=doc_index)
    elif retriever_type == "dpr":
        document_store.update_embeddings(retriever_dpr, index=doc_index)

    assert document_store.get_document_count(index=doc_index) == 3  # number of contexts
    assert document_store.get_label_count(index=label_index) == 20  # number of answers

    # eval retriever
    retriever_eval_results = eval_retriever(
        document_store=document_store,
        pipeline=p,
        top_k=3,
        label_index=label_index,
        doc_index=doc_index,
    )

    assert recall_expected == pytest.approx(retriever_eval_results["recall"], abs=1e-4)
    assert mrr_expected == pytest.approx(retriever_eval_results["mrr"], abs=1e-4)
