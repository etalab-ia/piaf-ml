from haystack.retriever.dense import DensePassageRetriever

dpr = DensePassageRetriever(
    document_store=None,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_fast_tokenizers=False,
)

dpr.train(
    data_dir="/home/pavel/code/piaf-ml/data/squad2dpr",
    train_filename="DPR_FR_train.json",
    dev_filename="DPR_FR_dev.json",
    test_filename=None,
    save_dir="/data/models/dpr/haystack_camembert_dpr",
    passage_encoder_save_dir="ctx_encoder",
    query_encoder_save_dir="query_encoder",
    batch_size=4,
    n_gpu=1,
)
