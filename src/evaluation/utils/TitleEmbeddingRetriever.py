from haystack.retriever.dense import EmbeddingRetriever

from typing import List

import numpy as np

from haystack import Document



class TitleEmbeddingRetriever(EmbeddingRetriever):
    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings of the titles for a list of passages. For this Retriever type: The same as calling .embed()

        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        texts = [d.meta['name'] for d in docs]

        return self.embed(texts)