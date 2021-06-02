parameters = {
    "k": [3],
    "retriever_type": ["google"],
    "squad_dataset": [
        "./test/samples/squad/tiny.json"
    ],  # data/evaluation-datasets/fquad_valid_with_impossible_fraction.json data/evaluation-datasets/testing_squad_format.json
    "filter_level": [None],
    "boosting": [1],
    "preprocessing": [False],
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [200],
    "split_respect_sentence_boundary": [True],
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
