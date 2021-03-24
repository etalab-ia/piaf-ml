parameters = {
    "k_retriever": [3],
    "squad_dataset": ["data/evaluation-datasets/besoindaide_PIAF.json"], # data/evaluation-datasets/fquad_valid_with_impossible_fraction.json data/evaluation-datasets/testing_squad_format.json
    "filter_level": [None],
    "preprocessing": [True],
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
