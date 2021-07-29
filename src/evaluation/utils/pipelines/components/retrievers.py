from deployment.roles.haystack.files.custom_component import \
       TitleEmbeddingRetriever, LabelElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from src.evaluation.utils.google_retriever import GoogleRetriever
from src.evaluation.utils.epitca_retriever import EpitcaRetriever

def bm25(document_store, top_k):
    return ElasticsearchRetriever(document_store=document_store, top_k = top_k)

def epitca(document_store):
    return EpitcaRetriever(document_store=document_store)

def google(document_store, google_retriever_website):
    return GoogleRetriever(document_store=document_store,
            website=google_retriever_website)

def sbert(document_store, retriever_model_version, gpu_available):
    return EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
            model_version=retriever_model_version,
            use_gpu=gpu_available,
            model_format="transformers",
            pooling_strategy="reduce_max",
            emb_extraction_layer=-1,
        )

def dpr(document_store, dpr_model_version, gpu_available):
    return DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="etalab-ia/dpr-question_encoder-fr_qa-camembert",
            passage_embedding_model="etalab-ia/dpr-ctx_encoder-fr_qa-camembert",
            model_version=dpr_model_version,
            infer_tokenizer_classes=True,
            use_gpu=gpu_available,
        )

def title(document_store, retriever_model_version, gpu_available, top_k):
    return TitleEmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
            model_version=retriever_model_version,
            use_gpu=gpu_available,
            model_format="transformers",
            pooling_strategy="reduce_max",
            emb_extraction_layer=-1,
            top_k = top_k,
        )

def label_elasticsearch_retriever(document_store, top_k,
        weight_when_document_found):
    return LabelElasticsearchRetriever(
            document_store = document_store,
            top_k = top_k,
            weight_when_document_found = weight_when_document_found)
