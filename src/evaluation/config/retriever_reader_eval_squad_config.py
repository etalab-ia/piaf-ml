parameters = {
    "k_retriever": [3,15,30,45,60],
    "k_reader": [3,5,10],
    "retriever_type": ["bm25"],
    "squad_dataset": ["./data/evaluation-datasets/full_spf_squad.json", "data/evaluation-datasets/fquad_eval.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "split_by": ["word"], #Can be "word", "sentence", or "passage"
    "split_length":[200,500,1000]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
