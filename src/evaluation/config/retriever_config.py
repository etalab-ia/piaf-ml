parameters = {
    "k": [5],
    "retriever_type": ["bm25"],
    "knowledge_base": ["./data/v12"],
    "test_dataset": ["./data/evaluation-datasets/407_question-fiche_anonym.csv"],
    "weighted_precision": [True],
    "filter_level": [None],
    # "elasticsearch_url": [("https://psorianom-elk-9149282307489525752-elastic-api.lab.sspcloud.fr", 443)],
    "elasticsearch_url": [("localhost", 9200)],
    "preprocessing": [False]
}
# rules:
# corpus and retriever type requires reloading ES indexing
# filtering requires >v10
#
