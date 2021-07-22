from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

def elasticsearch(elasticsearch_hostname, elasticsearch_port,
        similarity, embedding_dim, title_boosting_factor):

    return ElasticsearchDocumentStore(
            host=elasticsearch_hostname,
            port=elasticsearch_port,
            username="",
            password="",
            index="document_elasticsearch",
            search_fields=["name", "text"],
            create_index=False,
            embedding_field="emb",
            scheme="",
            embedding_dim=embedding_dim,
            excluded_meta_data=["emb"],
            similarity=similarity,
            custom_mapping=squad_mapping(embedding_dim, title_boosting_factor),
        )

def squad_mapping(embedding_dim, title_boosting_factor):
    return {
        "mappings": {
            "properties": {
                "name": {"type": "text", "boost": title_boosting_factor},
                "text": {"type": "text"},
                "emb": {"type": "dense_vector", "dims": embedding_dim},
            },
            "dynamic_templates": [
                {
                    "strings": {
                        "path_match": "*",
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword"},
                    }
                }
            ],
        },
        "settings": analyzer_default(),
    }

def analyzer_default():
    return {
        "analysis": {
            "filter": {
                "french_elision": {
                    "type": "elision",
                    "articles_case": True,
                    "articles": [
                        "l",
                        "m",
                        "t",
                        "qu",
                        "n",
                        "s",
                        "j",
                        "d",
                        "c",
                        "jusqu",
                        "quoiqu",
                        "lorsqu",
                        "puisqu",
                    ],
                },
                "french_stop": {"type": "stop", "stopwords": "_french_"},
                "french_stemmer": {"type": "stemmer", "language": "light_french"},
            },
            "analyzer": {
                "default": {
                    "tokenizer": "standard",
                    "filter": [
                        "french_elision",
                        "lowercase",
                        "french_stop",
                        "french_stemmer",
                    ],
                }
            },
        }
    }

