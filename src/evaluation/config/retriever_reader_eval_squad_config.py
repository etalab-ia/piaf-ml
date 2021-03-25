parameters = {
    "k_retriever": [3],
    "k_title_retriever" : [3], # must be present but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [1],
    "k_reader_total": [1],
    "retriever_type": ["title_bm25"], # Can be bm25, sbert, dpr or title_bm25
    "squad_dataset": ["./test/samples/squad/tiny.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "boosting" : [1], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
    "experiment_name": ["dev"]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
