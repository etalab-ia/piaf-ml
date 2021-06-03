parameters = {
    "k_retriever": [3],
    "squad_dataset": [
        "./clients/cnil/knowledge_base/besoindaide_PIAF_V5.json"
    ],  # data/evaluation-datasets/fquad_valid_with_impossible_fraction.json data/evaluation-datasets/testing_squad_format.json
    "filter_level": [None],
    "preprocessing": [True],
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
