from haystack.reader.transformers import TransformersReader

def transformers_reader(reader_model_version, gpu_id, k_reader_per_candidate):
    return TransformersReader(
        model_name_or_path="etalab-ia/camembert-base-squadFR-fquad-piaf",
        tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf",
        model_version=reader_model_version,
        use_gpu=gpu_id,
        top_k_per_candidate=k_reader_per_candidate,
    )

