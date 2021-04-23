parameters = {
    "k_retriever": [1],
    "k_title_retriever" : [10], # must be present, but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [5],
    "k_reader_total": [3],
    "retriever_type": ["bm25"], # Can be bm25, sbert, dpr, title or title_bm25
    "squad_dataset": ["data/evaluation-datasets/tiny.json"],
    "filter_level": [None],
    "preprocessing": [True, False],
    "boosting" : [1], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [10],
    "experiment_name": ["dev"]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
