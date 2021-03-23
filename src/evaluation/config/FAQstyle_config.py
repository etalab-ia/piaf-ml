parameters = {
    "k_retriever": [3],
    "retriever_type": ["sbert"],
    "squad_dataset": ["./test/samples/squad/faq.json"], # data/evaluation-datasets/fquad_valid_with_impossible_fraction.json data/evaluation-datasets/testing_squad_format.json
    "filter_level": [None],
    "preprocessing": [True],
    "split_by": ["word"], #Can be "word", "sentence", or "passage"
    "split_length":[200],
    "split_respect_sentence_boundary": [True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
