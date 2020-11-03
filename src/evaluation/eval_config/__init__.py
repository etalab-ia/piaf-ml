parameters = {
    "k": [5],
    "retriever_type": ['dense'],
    "knowledge_base": ["./data/v12"],
                       # "/home/pavel/code/piaf-ml/data/v10"],
    "test_dataset": ["./data/407_question-fiche_anonym.csv"],
    "weighted_precision": [True],
    "filter_level": [None],
    "lemma_preprocessing": [True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
