parmeters = {
    "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "retriever_type": ["sparse", "dense"],
    "knowledge_base": ["/home/pavel/code/piaf-ml/data/v11",
                       "/home/pavel/code/piaf-ml/data/v10"],
    "test_dataset": ["./data/25k_data/15082020-ServicePublic_QR_20170612_20180612_464_single_questions.csv"],
    "weighted_precision": [True, False],
    "filtering": [True, False]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
