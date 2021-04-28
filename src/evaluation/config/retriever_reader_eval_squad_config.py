parameters = {
    "k_retriever": [1,50],
    "k_title_retriever" : [1,50], # must be present, but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [1,20],
    "k_reader_total": [5],
    "reader_model_version": ["053b085d851196110d7a83d8e0f077d0a18470be"],
    "retriever_model_version": ["1a01b38498875d45f69b2a6721bf6fe87425da39"],
    "retriever_type": ["bm25","title_bm25"], # Can be bm25, sbert, dpr, title or title_bm25
    "squad_dataset": ["./clients/dila/knowledge_base/full_spf_squad.json"],
    "filter_level": [None],
    "preprocessing": [False],
    "boosting" : [1,10], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000000],
    "experiment_name": ["test_scikitoptimize"]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
