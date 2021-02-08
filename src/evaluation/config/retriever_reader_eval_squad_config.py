parameters = {
<<<<<<< HEAD
    "k_retriever": [3],
    "k_reader": [1],
    "retriever_type": ["dpr"],
    "squad_dataset": ["./data/evaluation-datasets/spf_squad.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [100],
    "split_respect_sentence_boundary": [True]
=======
    "k_retriever": [1,5,20,50,100],
    "k_reader": [1,5,20,50,100],
    "retriever_type": ["bm25"],
    "squad_dataset": ["./data/evaluation-datasets/full_spf_squad.json", "data/evaluation-datasets/fquad_eval.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "split_by": ["word"], #Can be "word", "sentence", or "passage"
    "split_length":[200,500,1000]
>>>>>>> master
}
