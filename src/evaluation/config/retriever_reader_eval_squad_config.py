parameters = {
    "k_retriever": [3],
    "k_title_retriever": [1],  # must be present, but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [20],
    "k_reader_total": [5],
    "threshold_score": [1.00],  # must be present, but only used when retriever_type == hot_reader
    "reader_model_version": ["053b085d851196110d7a83d8e0f077d0a18470be"],
    "retriever_model_version": ["fcd5c2bb3e3aa74cd765d793fb576705e4ea797e"],
    "dpr_model_version": ["v1.0"],
    "retriever_type": ["title"],  # Can be bm25, sbert, dpr, title or title_bm25
    "squad_dataset": ["./clients/cnil/knowledge_base/besoindaide_PIAF_V5.json"],
    "filter_level": [None],
    "preprocessing": [False],
    "boosting": [1],  # default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
}

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
