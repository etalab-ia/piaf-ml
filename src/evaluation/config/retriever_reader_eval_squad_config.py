parameters = {
    "k_retriever": [3],
    "k_title_retriever" : [1], # must be present, but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [20],
    "k_reader_total": [5],
    "reader_model_version": ["053b085d851196110d7a83d8e0f077d0a18470be"],
    "retriever_model_version": ["1a01b38498875d45f69b2a6721bf6fe87425da39"],
    "dpr_model_version": ["v1.0"],
    "retriever_type": ["title"], # Can be bm25, sbert, dpr, title or title_bm25
    "squad_dataset": ["./test/samples/squad/tiny.json"],
    "filter_level": [None],
    "preprocessing": [False],
    "boosting" : [1], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#

parameter_tuning_options = {
    # "experiment_name": "DILA_fullspfV1",
    "experiment_name": "test",

    # Tuning method alternatives:
    # - "optimization": use bayesian optimisation
    # - "grid_search"
    "tuning_method": "grid_search",

    # Additionnal options for the grid search method
    "use_cache": False,

    # Additionnal options for the optimization method
    "optimization_ncalls": 10,
}
