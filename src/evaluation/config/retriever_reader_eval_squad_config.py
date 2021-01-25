parameters = {
    "k_retriever": [3,15],
    "k_reader": [1,3,5],
    "retriever_type": ["bm25"],
    "squad_dataset": ["./data/evaluation-datasets/spf_squad.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "split_by": ["word"], #Can be "word", "sentence", or "passage"
    "split_length":[100,200,500,1000],
    "split_respect_sentence_boundary": [True]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
