parameters = {
    "k": [5],
    "retriever_type": ["bm25"],
    "knowledge_base": ["./data/v12"],
    "test_dataset": ["./data/evaluation-datasets/407_question-fiche_anonym.csv"],
    "weighted_precision": [True],
    "filter_level": [None],
    "preprocessing": [False, True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
