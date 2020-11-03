parameters = {
    "k": [5],
    "retriever_type": ['sparse'],
    "knowledge_base": ["./data/v11"],
    "test_dataset": ["./data/407_question-fiche_anonym.csv"],
    "weighted_precision": [True],
    "filter_level": ['theme'],
    "lemma_preprocessing": [False],
    "dual_retriever_top_k": [50]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
