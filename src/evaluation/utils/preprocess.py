import json
import logging
from typing import List, Tuple, Union

from haystack import Document, Label
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


def add_eval_data_from_file(
    filename: str, retriever_emb: BaseRetriever, max_docs: Union[int, bool] = None
) -> Tuple[List[Document], List[Label]]:
    """
    Read Documents + Labels from a SQuAD-style file. Document and Labels can then be indexed to the DocumentStore and be
    used for evaluation.

    :param retriever_emb: the dense retriever to embed the text 
    :param filename: Path to file in SQuAD format 
    :param max_docs: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents. 
    :return: (List of Documents, List of Labels)
    """

    docs: List[Document] = []
    labels = []

    with open(filename, "r") as file:
        data = json.load(file)
        if "title" not in data["data"][0]:
            logger.warning(
                f"No title information found for documents in QA file: {filename}"
            )
        for document in data["data"]:
            if max_docs:
                if len(docs) > max_docs:
                    break
            # get all extra fields from document level (e.g. title)
            meta_doc = {
                k: v for k, v in document.items() if k not in ("paragraphs", "title")
            }
            for paragraph in document["paragraphs"]:
                if max_docs:
                    if len(docs) > max_docs:
                        break
                cur_meta = {"name": document.get("title", None)}
                # all other fields from paragraph level
                meta_paragraph = {
                    k: v for k, v in paragraph.items() if k not in ("qas", "context")
                }
                cur_meta.update(meta_paragraph)
                # meta from parent document
                cur_meta.update(meta_doc)

                text = paragraph["context"]
                assert retriever_emb is not None
                embedding = retriever_emb.embed(text)[0]
                # Create Document
                cur_doc = Document(
                    text=paragraph["context"], embedding=embedding, meta=cur_meta
                )

                docs.append(cur_doc)

                # Get Labels
                for qa in paragraph["qas"]:
                    if len(qa["answers"]) > 0:
                        for answer in qa["answers"]:
                            label = Label(
                                question=qa["question"],
                                answer=answer["text"],
                                is_correct_answer=True,
                                is_correct_document=True,
                                document_id=cur_doc.id,
                                offset_start_in_doc=answer["answer_start"],
                                no_answer=qa["is_impossible"],
                                origin="gold_label",
                            )
                            labels.append(label)
                    else:
                        label = Label(
                            question=qa["question"],
                            answer="",
                            is_correct_answer=True,
                            is_correct_document=True,
                            document_id=cur_doc.id,
                            offset_start_in_doc=0,
                            no_answer=qa["is_impossible"],
                            origin="gold_label",
                        )
                        labels.append(label)
        return docs, labels
