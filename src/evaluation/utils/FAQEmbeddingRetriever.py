from haystack.retriever.dense import EmbeddingRetriever

import logging
from typing import List, Union, Tuple, Optional
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from haystack.document_store.base import BaseDocumentStore
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack import Document
from haystack.retriever.base import BaseRetriever

from farm.infer import Inferencer
from farm.modeling.tokenization import Tokenizer
from farm.modeling.language_model import LanguageModel
from farm.modeling.biadaptive_model import BiAdaptiveModel
from farm.modeling.prediction_head import TextSimilarityHead
from farm.data_handler.processor import TextSimilarityProcessor
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.dataloader import NamedDataLoader
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from torch.utils.data.sampler import SequentialSampler



class FAQEmbeddingRetriever(EmbeddingRetriever):
    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of passages. For this Retriever type: The same as calling .embed()

        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        texts = [d.meta['name'] for d in docs]

        return self.embed(texts)