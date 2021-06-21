parameters = {
    "k_retriever": [1],
    "k_title_retriever" : [1],# must be present, but only used when retriever_type == title_bm25, or hot_reader
    "k_reader_per_candidate": [20],
    "k_reader_total": [5],
    "threshold_score": [1.00],# must be present, but only used when retriever_type == hot_reader
    "reader_model_version": ["053b085d851196110d7a83d8e0f077d0a18470be"],
    "retriever_model_version": ["1a01b38498875d45f69b2a6721bf6fe87425da39"],
    "retriever_type": ["title_bm25","bm25"], # Can be bm25, sbert, dpr, title, hot_reader, or title_bm25
    "squad_dataset": [
        "./clients/dila/knowledge_base/squad.json",
    ],
    "filter_level": [None],
    "preprocessing": [False],
    "boosting" : [1], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
    "experiment_name": ["hot_reader_dila"]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
