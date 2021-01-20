parameters = {
    "k": [3],
    "retriever_type": ["bm25"],
    "squad_dataset": ["./test/samples/squad/tiny.json"], # data/evaluation_datasets/fquad_valid_with_impossible_fraction.json data/evaluation_datasets/testing_squad_format.json
    "filter_level": [None],
    "preprocessing": [True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
