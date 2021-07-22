parameters = {
    "k": [1, 3],
    "retriever_type": ['epitca', 'bm25'],
    "google_retriever_website": ['service-public.fr'],
    "squad_dataset": [
        "./clients/cnil/knowledge_base/squad.json"
    ],  # data/evaluation-datasets/fquad_valid_with_impossible_fraction.json data/evaluation-datasets/testing_squad_format.json
    # Path to the Epitca performance file or None. Needed when using the
    # retriever_type 'epitca'.
    "epitca_perf_file": ["./clients/cnil/knowledge_base/raw_and_preparation/epitca_perf_V2.json"],
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
