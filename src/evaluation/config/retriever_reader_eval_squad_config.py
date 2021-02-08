parameters = {
    "k_retriever": [3],
    "k_reader": [1],
    "retriever_type": ["dpr"],
    "squad_dataset": ["/home/pavel/code/piaf-ml/data/evaluation-datasets/small_modif.json"],#["./data/evaluation-datasets/spf_squad.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [100],
    "split_respect_sentence_boundary": [True]

}
