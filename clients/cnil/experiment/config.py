parameters = {
    "weight_when_document_found": [99],
    "k_bm25_retriever": [5],
    "k_title_retriever" : [5],
    "k_label_retriever" : [5],
    "ks_retriever": [[1, 1, 1]],
    "k_reader_per_candidate": [20],
    "k_reader_total": [5],
    "threshold_score": [1.00],# must be present, but only used when retriever_type == hot_reader
    "reader_model_version": ["053b085d851196110d7a83d8e0f077d0a18470be"],
    "retriever_model_version": ["fcd5c2bb3e3aa74cd765d793fb576705e4ea797e"],
    "squad_dataset": ["./test/samples/squad/tiny.json"],
    "boosting" : [1], #default to 1
    "context_window_size": [200],
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#

parameter_tuning_options = {
    # "experiment_name": "DILA_fullspfV1",
    "experiment_name": "cnil",

    # Tuning method alternatives:
    # - "optimization": use bayesian optimisation
    # - "grid_search"
    "tuning_method": "grid_search",

    # Additionnal options for the grid search method
    "use_cache": False,
}
