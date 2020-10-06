parmeters = {
    "k": [1, 2, 3],
    "retriever_type": ["sparse"],
    "knowledge_base": ["/home/pavel/code/piaf-ml/data/v11"],
                       # "/home/pavel/code/piaf-ml/data/v10"],
    "test_dataset": ["./data/25k_data/15082020-ServicePublic_QR_20170612_20180612_464_single_questions.csv"],
    "weighted_precision": [True],
    "filter_level": [None, "theme", "dossier"]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#