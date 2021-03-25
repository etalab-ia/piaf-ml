ANALYZER_DEFAULT = {
    "analysis": {
        "filter": {
            "french_elision": {
                "type": "elision",
                "articles_case": True,
                "articles": [
                    "l", "m", "t", "qu", "n", "s",
                    "j", "d", "c", "jusqu", "quoiqu",
                    "lorsqu", "puisqu"
                ]
            },
            "french_stop": {
                "type": "stop",
                "stopwords": "_french_"
            },
            "french_stemmer": {
                "type": "stemmer",
                "language": "light_french"
            }
        },
        "analyzer": {
            "default": {
                "tokenizer": "standard",
                "filter": [
                    "french_elision",
                    "lowercase",
                    "french_stop",
                    "french_stemmer"
                ]
            }
        }
    }
}

SBERT_MAPPING = {
    "mappings": {
        "properties": {
            "link": {
                "type": "keyword"
            },
            "name": {
                "type": "keyword"
            },
            "question_sparse": {
                "type": "text"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 512
            },
            "text": {
                "type": "text"
            },
            "theme": {
                "type": "keyword"
            },
            "dossier": {
                "type": "keyword"
            }
        }},
    "settings": ANALYZER_DEFAULT
}

DPR_MAPPING = {
    "mappings": {
        "properties": {
            "link": {
                "type": "keyword"
            },
            "name": {
                "type": "keyword"
            },
            "question_sparse": {
                "type": "text"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 768
            },
            "text": {
                "type": "text"
            },
            "theme": {
                "type": "keyword"
            },
            "dossier": {
                "type": "keyword"
            }
        }},
    "settings": ANALYZER_DEFAULT
}

SPARSE_MAPPING = {
    "mappings": {
        "properties": {
            "question_sparse": {
                "type": "text",
            },
            "text": {
                "type": "text"
            },
            "theme": {
                "type": "keyword"
            },
            "dossier": {
                "type": "keyword"
            }
        }
    },
    "settings": ANALYZER_DEFAULT
}

SQUAD_MAPPING = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "text": {"type": "text"},
            "emb": {"type": "dense_vector", "dims": 512}
        },
        "dynamic_templates": [
            {
                "strings": {
                    "path_match": "*",
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"}}}
        ],
    },
    "settings": ANALYZER_DEFAULT
}
