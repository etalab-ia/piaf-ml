parameters = {
    "k": [5],
    "retriever_type": ["sbert"],
    "knowledge_base": ["./data/v14"],
    "test_dataset": ["./data/evaluation-datasets/407_question-fiche_anonym.csv"],
    "weighted_precision": [True],
    "filter_level": [None],
    "preprocessing": [True],
    "use_cache": [True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
