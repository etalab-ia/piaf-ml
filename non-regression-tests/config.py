import os

parameter_tuning_options = {
    "experiment_name": "non-regression-tests",

    # Tuning method alternatives:
    # - "optimization": use bayesian optimisation
    # - "grid_search"
    "tuning_method": "grid_search",

    # Additionnal options for the grid search method
    "use_cache": False,

    # Additionnal options for the optimization method
    "optimization_ncalls": 10,
}

parameters_fquad = {
    "k_retriever": [5],
    "k_title_retriever" : [1], # must be present, but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [20],
    "k_reader_total": [10],
    "reader_model_version": ["053b085d851196110d7a83d8e0f077d0a18470be"],
    "retriever_model_version": ["1a01b38498875d45f69b2a6721bf6fe87425da39"],
    "dpr_model_version": ["v1.0"],
    "retriever_type": ["bm25"], # Can be bm25, sbert, dpr, title or title_bm25
    "squad_dataset": [
        os.getenv("DATA_DIR") + "/non-regression-tests/fquad_dataset.json"
    ],
    "filter_level": [None],
    "preprocessing": [False],
    "boosting" : [1], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
}

# A dictionnary specifying the criteria a test result must pass. Keys are
# metrics names and keys are predicates on the corresponding metric which must
# return true if the value is satisfying.
pass_criteria_fquad = {
    "reader_topk_accuracy_has_answer": 
        # metric ~= 0.747 +/- 1%
        lambda metric: abs(metric / 0.747 - 1) < 0.01
}

parameters_dila = {
    "k_retriever": [1],
    "k_title_retriever" : [1], # must be present, but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [20],
    "k_reader_total": [10],
    "reader_model_version": ["053b085d851196110d7a83d8e0f077d0a18470be"],
    "retriever_model_version": ["1a01b38498875d45f69b2a6721bf6fe87425da39"],
    "dpr_model_version": ["v1.0"],
    "retriever_type": ["bm25"], # Can be bm25, sbert, dpr, title or title_bm25
    "squad_dataset": [
        os.getenv("SRC_DIR") + "/piaf-ml/clients/dila/knowledge_base/squad.json"],
    "filter_level": [None],
    "preprocessing": [False],
    "boosting" : [1], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
}

# A dictionnary specifying the criteria a test result must pass. Keys are
# metrics names and keys are predicates on the corresponding metric which must
# return true if the value is satisfying.
pass_criteria_dila = {
    "reader_topk_accuracy_has_answer":
        # metric ~= 0.427 +/- 1%
        lambda metric: abs(metric / 0.427 - 1) < 0.01
}


tests = [
    (parameters_fquad, parameter_tuning_options, pass_criteria_fquad),
    (parameters_dila, parameter_tuning_options, pass_criteria_dila),
]
