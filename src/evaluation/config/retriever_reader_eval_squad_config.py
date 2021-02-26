parameters = {
    "k_retriever": [1],
    "k_reader": [1],
    "k_display": [1],
    "retriever_type": ["bm25"],
    "squad_dataset": ["./data/evaluation-datasets/full_spf_squad.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "split_by": ["word"], #Can be "word", "sentence", or "passage"
    "split_length":[1000]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
