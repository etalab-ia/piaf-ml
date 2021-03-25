parameters = {
    "k_retriever": [3],
    "k_reader_per_candidate": [1],
    "k_reader_total": [1],
    "retriever_type": ["bm25"],
    "squad_dataset": ["./data/evaluation-datasets/full_spf_squad.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "boosting" : [1, 2], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
    "experiment_name": ["dev"]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
