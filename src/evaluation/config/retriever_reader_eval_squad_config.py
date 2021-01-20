parameters = {
    "k_retriever": [1],
    "k_reader": [1],
    "retriever_type": ["bm25"],
    "squad_dataset": ["./data/evaluation_datasets/fquad_eval.json"],
    "filter_level": [None],
    "preprocessing": [True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
