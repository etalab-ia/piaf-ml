parameters = {
    "k": [5],
    "retriever_type": ["sbert"],
    "squad_dataset": ["./data/evaluation-datasets/fquad_valid_with_impossible_fraction.json"],
    "filter_level": [None],
    "preprocessing": [False],
    "mapping_config": ['squad']
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
