parameters = {
    "k": [1, 3],
    # "retriever_type": ["bm25", "sbert", "google", "epitca"],
    "retriever_type": ["bm25"],
    "retriever_model_version": ["1a01b38498875d45f69b2a6721bf6fe87425da39"],
    "google_retriever_website": ['service-public.fr'],
    # "squad_dataset": ["./clients/cnil/knowledge_base/squad.json"],
    "squad_dataset": ["./test/samples/squad/tiny.json"],
    # Path to the Epitca performance file or None. Needed when using the
    # retriever_type 'epitca'.
    #"epitca_perf_file": ["./clients/cnil/knowledge_base/raw_and_preparation/epitca_perf_V2.json"],
    "epitca_perf_file": [None],
    "filter_level": [None],
    "boosting": [1],
    "preprocessing": [False],
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [200],
    "split_respect_sentence_boundary": [True],
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
