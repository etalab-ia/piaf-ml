parameters = {
    "k": [5],   
    "retriever_type": ['both','sparse','dense'],
    "knowledge_base": ["./data/v11"],
                       # "/home/pavel/code/piaf-ml/data/v10"],
    "test_dataset": ["./data/407_question-fiche_anonym.csv"],
    "weighted_precision": [False],
    "filter_level": [None],
    "lemma_preprocessing": [False]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
