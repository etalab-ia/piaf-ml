import pytest
from pathlib import Path
from collections import defaultdict

from haystack.document_store.base import BaseDocumentStore
from haystack.pipeline import Pipeline
from haystack.retriever.sparse import ElasticsearchFilterOnlyRetriever
from haystack.reader.transformers import TransformersReader

from src.evaluation.utils.utils_eval import full_eval_retriever_reader
from deployment.roles.haystack.files.custom_component import \
    MergeOverlappingAnswers, merge_strings



def test_merge_strings():
    assert merge_strings("a black", "black coffee", 0.1) == ("a black coffee", 2)
    assert merge_strings("black coffee", "a black", 0.1) == ("a black coffee", -2)
    assert merge_strings("a black coffee", "black", 0.1) == ("a black coffee", 2)
    assert merge_strings("a black coffee", "black", 1) == ("a black coffee", 2)
    assert merge_strings("a black coffee", "a black coffee", 1) == ("a black coffee", 0)
    assert merge_strings("a black coffee", "black coffee,", 1) == ("", 0)
    assert merge_strings("a black coffee", " with milk", 0.1) == ("", 0)
    assert merge_strings("a black coffee", " with milk", 0) == ("a black coffee with milk", 14)
    assert merge_strings("a black coffee", "", 0) == ("", 0)
    assert merge_strings("a black coffee", "", 0.1) == ("", 0)
    assert merge_strings("a coffee is my first thing in the morning", "morning or evening", 0.5) == ("", 0)
    assert merge_strings("a coffee is my first thing in the morning", "in the morning", 0.5) == ("a coffee is my first thing in the morning", 27)


@pytest.mark.elasticsearch
def test_no_duplicate_answers(document_store: BaseDocumentStore):
    doc_index = "document"
    label_index = "label"

    retriever = ElasticsearchFilterOnlyRetriever(document_store=document_store)

    reader = TransformersReader(
        model_name_or_path="etalab-ia/camembert-base-squadFR-fquad-piaf",
        tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf",
        use_gpu=-1,
        top_k_per_candidate=4,
        context_window_size=200,
        top_k=5,
        return_no_answers=False
    )

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=reader, name="Reader", inputs=['Retriever'])
    p.add_node(component=MergeOverlappingAnswers(0.75, 0.25), name="MergeOverlappingAnswers", inputs=["Reader"])

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(
        filename=Path("./test/samples/squad/duplicate_answers.json").as_posix(),
        doc_index=doc_index,
        label_index=label_index,
    )

    res = p.run(query="c'est quoi un cookie ?", top_k_retriever = 10, top_k_reader = 5)

    # There must be only one answer that contains the text "petit fichier informatique"
    answers = [ans for ans in res["answers"] if ans["answer"] and ans["answer"].find("petit fichier informatique") != -1]
    assert len(answers) == 1

    assert answers[0]["offset_start"] == 14
    assert answers[0]["offset_end"] == 56

    # There should be other answers unrelated to "petit fichier informatique" that didn't get merged with the others.
    assert len(res["answers"]) > 1

    # clean up
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
