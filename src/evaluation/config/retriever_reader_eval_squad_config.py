parameters = {
    "k_retriever": [10],
    "k_reader": [3],
    "retriever_type": ["sbert"],
    "squad_dataset": ["./data/evaluation-datasets/fquad_eval.json"],
    "filter_level": [None],
    "preprocessing": [True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
