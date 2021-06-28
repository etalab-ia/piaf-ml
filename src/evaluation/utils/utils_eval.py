import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from haystack.document_store.base import BaseDocumentStore
from haystack.eval import eval_counts_reader, calculate_reader_metrics, \
    _count_no_answer, _calculate_f1, _count_overlap, _count_exact_match, \
    EvalDocuments, EvalAnswers
from haystack.pipeline import Pipeline
from tqdm import tqdm

logger = logging.getLogger()


def save_results(result_file_path: Path, results_list: Union[Dict, List[Dict]]):
    if not isinstance(results_list, list):
        results_list = [results_list]

    df_results = pd.DataFrame(results_list)
    if result_file_path.exists():
        df_old = pd.read_csv(result_file_path)
        df_results = pd.concat([df_old, df_results])
    else:
        if not result_file_path.parent.exists():
            result_file_path.parent.mkdir()
    with open(result_file_path.as_posix(), "w") as filo:
        df_results.to_csv(filo, index=False)


def eval_retriever(
        document_store: BaseDocumentStore,
        pipeline: Pipeline,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k: int = 10,
        return_preds: bool = False,
        question_label_dict_list=None,
        get_doc_id=lambda doc: doc.id,
) -> dict:
    """
    :param question_label_dict_list: A list [{"query": ..., "gold_ids": [id1, ...]}, 
    ...] where the query fields are questions to ask for evaluation and the gold_ids are
    lists of document ids expected as answers. If None, the questions and
    associated answers are fetched from the document_store gold_label index.
    """

    if question_label_dict_list == None:

        # Extract all questions for evaluation
        filters = {"origin": [label_origin]}

        labels = document_store.get_all_labels_aggregated(
            index=label_index, filters=filters
        )
        # Collect questions and corresponding answers/document_ids in a dict
        question_label_dict_list = []
        for label in labels:
            if not label.question:
                continue
            deduplicated_doc_ids = list(set([str(x) for x in label.multiple_document_ids]))
            question_label_dict = {
                "query": label.question,
                "gold_ids": deduplicated_doc_ids,
            }
            question_label_dict_list.append(question_label_dict)

    retrieved_docs_list = [
        pipeline.run(query=question["query"], top_k_retriever=top_k, index=doc_index)
        for question in question_label_dict_list
    ]

    metrics = get_retriever_metrics(retrieved_docs_list, question_label_dict_list, get_doc_id)

    logger.info(
        (
            f"For {metrics['correct_retrievals']} out of {metrics['number_of_questions']} questions ({metrics['recall']:.2%}), the answer was in"
            f" the top-{top_k} candidate passages selected by the retriever."
        )
    )

    return metrics


def get_retriever_metrics(retrieved_docs_list, question_label_dict_list,
                          get_doc_id=lambda doc: doc.id):
    correct_retrievals = 0
    summed_avg_precision = 0.0
    summed_reciprocal_rank = []

    for question, retrieved_docs in tqdm(
            zip(question_label_dict_list, retrieved_docs_list)
    ):
        gold_ids = question["gold_ids"]
        number_relevant_docs = len(gold_ids)
        # check if correct doc in retrieved docs
        found_relevant_doc = False
        relevant_docs_found = 0
        current_avg_precision = 0.0
        for doc_idx, doc in enumerate(retrieved_docs["documents"]):
            if str(get_doc_id(doc)) in gold_ids:
                relevant_docs_found += 1
                if not found_relevant_doc:
                    correct_retrievals += 1
                    summed_reciprocal_rank.append(1 / (doc_idx + 1))
                current_avg_precision += relevant_docs_found / (doc_idx + 1)
                found_relevant_doc = True
                if relevant_docs_found == number_relevant_docs:
                    break
        if found_relevant_doc:
            all_relevant_docs = len(set(gold_ids))
            summed_avg_precision += current_avg_precision / all_relevant_docs

    # Metrics
    number_of_questions = len(question_label_dict_list)
    recall = correct_retrievals / number_of_questions
    mean_reciprocal_rank = sum(summed_reciprocal_rank) / number_of_questions
    mean_avg_precision = summed_avg_precision / number_of_questions

    metrics = {
        "recall": recall,
        "map": mean_avg_precision,
        "mrr": mean_reciprocal_rank,
        "correct_retrievals": correct_retrievals,
        "number_of_questions": number_of_questions,
    }

    return metrics


def eval_retriever_reader(
        document_store: BaseDocumentStore,
        pipeline: Pipeline,
        k_retriever: int,
        k_reader_total: int,
        label_index: str = "label",
        label_origin: str = "gold_label",
):
    """
    Performs evaluation on evaluation documents in the DocumentStore. Returns a dict containing the following metrics:

          - "EM": Proportion of exact matches of predicted answers with their corresponding correct answers
          - "f1": Average overlap between predicted answers and their corresponding correct answers
          - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

    :param pipeline:
    :param document_store: DocumentStore containing the evaluation documents
    :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
    :param label_index: Index/Table name where labeled questions are stored
    :param doc_index: Index/Table name where documents that are used for evaluation are stored
    """

    # extract all questions for evaluation
    filters = {"origin": [label_origin]}
    # labels = document_store.get_all_labels(index=label_index, filters=filters)
    labels_agg = document_store.get_all_labels_aggregated(
        index=label_index, filters=filters
    )
    labels_agg = [label for label in labels_agg if label.question]

    questions = [label.question for label in labels_agg]
    predicted_answers_list = []
    for q in questions:
        answer = pipeline.run(
            query=q, top_k_retriever=k_retriever, top_k_reader=k_reader_total
        )
        predicted_answers_list.append(answer)

    assert len(questions) == len(predicted_answers_list), (
        f"Number of questions is not the same number of predicted" f"answers"
    )
    # quick renaming fix to match with haystack.eval.eval_counts_reader, this might be due to preprocessing
    # TODO : check this
    for predicted_answers in predicted_answers_list:
        for answer in predicted_answers["answers"]:
            answer["offset_start_in_doc"] = answer["offset_start"]
            answer["offset_end_in_doc"] = answer["offset_end"]

    metric_counts = {}
    metric_counts["correct_no_answers_top1"] = 0
    metric_counts["correct_no_answers_topk"] = 0
    metric_counts["correct_readings_topk"] = 0
    metric_counts["exact_matches_topk"] = 0
    metric_counts["summed_f1_topk"] = 0
    metric_counts["correct_readings_top1"] = 0
    metric_counts["correct_readings_top1_has_answer"] = 0
    metric_counts["correct_readings_topk_has_answer"] = 0
    metric_counts["summed_f1_top1"] = 0
    metric_counts["summed_f1_top1_has_answer"] = 0
    metric_counts["exact_matches_top1"] = 0
    metric_counts["exact_matches_top1_has_answer"] = 0
    metric_counts["exact_matches_topk_has_answer"] = 0
    metric_counts["summed_f1_topk_has_answer"] = 0
    metric_counts["number_of_no_answer"] = 0
    for question, predicted_answers in zip(labels_agg, predicted_answers_list):
        metric_counts = eval_counts_reader(question, predicted_answers, metric_counts)
    metrics = calculate_reader_metrics(metric_counts, len(predicted_answers_list))
    metrics.update(metric_counts)
    return metrics


class PiafEvalRetriever(EvalDocuments):
    """
    This is a pipeline node that should be placed after a Retriever in order to assess its performance. Performance
    metrics are stored in this class and updated as each sample passes through it.
    To extract the metrics in a dict form use EvalRetriever.get_metrics().
    """

    def __init__(self, debug: bool = False, open_domain: bool = False):
        super().__init__(debug, open_domain)

        self.summed_avg_precision = 0.0
        self.summed_reciprocal_rank = 0.0
        self.map = 0.0,  # mean_average_precision
        self.mrr = 0.0  # mean_reciprocal_rank

    def run(self, documents, labels: dict, **kwargs):
        """Run this node on one sample and its labels"""
        self.query_count += 1
        retriever_labels = labels["retriever"]

        if self.open_domain:

            if retriever_labels.no_answer:
                self.no_answer_count += 1
                correct_retrieval = 1
                self.summed_reciprocal_rank += 1
                self.summed_avg_precision += 1
                if not self.no_answer_warning:
                    self.no_answer_warning = True
                    logger.warning("There seem to be empty string labels in the dataset suggesting that there "
                                   "are samples with is_impossible=True. "
                                   "Retrieval of these samples is always treated as correct.")
            # If there are answer span annotations in the labels
            else:
                self.has_answer_count += 1
                correct_retrieval = self.is_correctly_retrieved(retriever_labels, documents)
                self.has_answer_correct += int(correct_retrieval)
                self.has_answer_recall = self.has_answer_correct / self.has_answer_count

        else:
            # call correct_retrievals_count if any document found we consider the retreiver operation correct
            correct_retrieval = self.is_correctly_retrieved(retriever_labels, documents)

        # update correct_retrieval_count
        self.correct_retrieval_count += correct_retrieval
        # update metrics
        self.recall = self.correct_retrieval_count / self.query_count
        self.map = self.summed_avg_precision / self.query_count
        self.mrr = self.summed_reciprocal_rank / self.query_count

        # if debug mode
        if self.debug:
            self.log.append(
                {"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs})

        # returns the required elements for the next node in the pipeline
        return {"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs}, "output_1"

    def is_correctly_retrieved(self, retriever_labels, predictions):
        """
        This function takes retriever_labels Multilabel object as well as the preedicted documents to update metric counts
        
        """
        relevant_docs_found = 0
        found_relevant_doc = False
        current_avg_precision = 0.0

        # extract the label document_ids (true relevant documents) and remove duplicated documents
        label_ids = list(set(retriever_labels.multiple_document_ids))

        if self.open_domain:
            for doc_idx, doc in enumerate(predictions):
                label_found = False
                for label in retriever_labels.multiple_answers:
                    # open domain : checks if answer in predicted document
                    if label.lower() in doc.text.lower() and not label_found:
                        label_found = True
                        relevant_docs_found += 1
                        if not found_relevant_doc:
                            # update summed_reciprocal_rank only for the first relevant predicted document
                            self.summed_reciprocal_rank += 1 / (doc_idx + 1)
                        # update avg_precision each time a new relevant doc is found
                        current_avg_precision += relevant_docs_found / (doc_idx + 1)
                        found_relevant_doc = True

            if found_relevant_doc:
                all_relevant_docs = len(label_ids)
                self.summed_avg_precision += current_avg_precision / all_relevant_docs
            return found_relevant_doc

        else:
            # check if the predicted document are relevant
            for doc_idx, doc in enumerate(predictions):
                # close domain checks if predicted document's ID in docuemt label ID
                if doc.id in label_ids:
                    # if predicted is relevant update count
                    relevant_docs_found += 1
                    if not found_relevant_doc:
                        # update summed_reciprocal_rank only for the first relevant predicted document
                        self.summed_reciprocal_rank += 1 / (doc_idx + 1)
                    # update avg_precision each time a new relevant doc is found
                    current_avg_precision += relevant_docs_found / (doc_idx + 1)
                    found_relevant_doc = True
            # if found relevant doc update summed_avg_precision
            if found_relevant_doc:
                all_relevant_docs = len(label_ids)
                self.summed_avg_precision += current_avg_precision / all_relevant_docs
            # returns true if relevant doc is found
            return found_relevant_doc

    def get_metrics(self):
        return {
            "recall": self.recall,
            "map": self.map,
            "mrr": self.mrr
        }


class PiafEvalReader(EvalAnswers):
    """
    This is a pipeline node that should be placed after a Reader in order to assess the performance of the Reader
    To extract the metrics in a dict form use EvalReader.get_metrics().
    """

    def __init__(self):

        super().__init__(debug=True, open_domain=False)

        self.metric_counts = {
            "correct_no_answers_top1": 0,
            "correct_no_answers_topk": 0,
            "correct_readings_topk": 0,
            "exact_matches_topk": 0,
            "summed_f1_topk": 0,
            "correct_readings_top1": 0,
            "correct_readings_top1_has_answer": 0,
            "correct_readings_topk_has_answer": 0,
            "summed_f1_top1": 0,
            "summed_f1_top1_has_answer": 0,
            "exact_matches_top1": 0,
            "exact_matches_top1_has_answer": 0,
            "exact_matches_topk_has_answer": 0,
            "summed_f1_topk_has_answer": 0,
            "number_of_no_answer": 0
        }

    def run(self, labels, answers, **kwargs):
        """Run this node on one sample and its labels"""

        self.query_count += 1

        multi_labels = labels["reader"]
        predictions = answers

        if multi_labels.no_answer:

            self.metric_counts['number_of_no_answer'] += 1
            self.metric_counts = _count_no_answer(predictions, self.metric_counts)
            if predictions:
                if self.debug:
                    self.log.append({"predictions": predictions,
                                     "gold_labels": multi_labels,
                                     "top_1_no_answer": int(predictions[0] == ""),
                                     })

        else:
            found_answer = False
            found_em = False
            best_f1 = 0

            for answer_idx, answer in enumerate(predictions):

                if answer["document_id"] in multi_labels.multiple_document_ids:
                    gold_spans = [{"offset_start": multi_labels.multiple_offset_start_in_docs[i],
                                   "offset_end": multi_labels.multiple_offset_start_in_docs[i] + len(
                                       multi_labels.multiple_answers[i]),
                                   "doc_id": multi_labels.multiple_document_ids[i]} for i in
                                  range(len(multi_labels.multiple_answers))]

                    predicted_span = {"offset_start": answer["offset_start"],
                                      "offset_end": answer["offset_end"],
                                      "doc_id": answer["document_id"]}

                    best_f1_in_gold_spans = 0
                    for gold_span in gold_spans:
                        if gold_span["doc_id"] == predicted_span["doc_id"]:
                            # check if overlap between gold answer and predicted answer
                            if not found_answer:
                                self.metric_counts, found_answer = _count_overlap(gold_span, predicted_span,
                                                                                  self.metric_counts,
                                                                                  answer_idx)  # type: ignore

                            # check for exact match
                            if not found_em:
                                self.metric_counts, found_em = _count_exact_match(gold_span, predicted_span,
                                                                                  self.metric_counts,
                                                                                  answer_idx)  # type: ignore

                            # calculate f1
                            current_f1 = _calculate_f1(gold_span, predicted_span)  # type: ignore
                            if current_f1 > best_f1_in_gold_spans:
                                best_f1_in_gold_spans = current_f1
                    # top-1 f1
                    if answer_idx == 0:
                        self.metric_counts["summed_f1_top1"] += best_f1_in_gold_spans
                        self.metric_counts["summed_f1_top1_has_answer"] += best_f1_in_gold_spans
                    if best_f1_in_gold_spans > best_f1:
                        best_f1 = best_f1_in_gold_spans

                if found_em:
                    break

            self.metric_counts["summed_f1_topk"] += best_f1
            self.metric_counts["summed_f1_topk_has_answer"] += best_f1

            if self.debug:
                self.log.append({"predictions": predictions,
                                 "gold_labels": multi_labels,
                                 "top_k_f1": self.metric_counts["summed_f1_topk"] / self.query_count,
                                 "top_k_em": self.metric_counts["exact_matches_topk"] / self.query_count
                                 })

        return {**kwargs}, "output_1"

    def get_metrics(self):
        metrics = calculate_reader_metrics(self.metric_counts, self.query_count)
        metrics.update(self.metric_counts)
        return (metrics)


def full_eval_retriever_reader(
        document_store: BaseDocumentStore,
        pipeline: Pipeline,
        k_retriever: int,
        k_reader_total: int,
        label_index: str = "label",
        label_origin: str = "gold_label",
):
    """
    Performs retriever/reader evaluation on evaluation documents in the DocumentStore.
    The role of this function is - prepare data format for evaluation pipeline
                                 - performs a single run (all the aueries) on the pipeline
    the function does not return evaluation results as they are stored on EvalRetriever and EvalRetriever Objects 

    :param pipeline:
    :param document_store: DocumentStore containing the evaluation documents
    :param label_index: Index/Table name where labeled questions are stored
    :param doc_index: Index/Table name where documents that are used for evaluation are stored
    """
    filters = {"origin": [label_origin]}
    labels = document_store.get_all_labels_aggregated(index=label_index, filters=filters)
    labels = [label for label in labels if label.question]
    q_to_l_dict = {
        l.question: {
            "retriever": l,
            "reader": l
        } for l in labels
    }

    for q, l in q_to_l_dict.items():
        pipeline.run(
            query=q,
            top_k_retriever=k_retriever,
            labels=l,
            top_k_reader=k_reader_total,
        )


def eval_titleQA_pipeline(
        document_store: BaseDocumentStore,
        pipeline: Pipeline,
        k_retriever: int,
        label_index: str = "label",
        label_origin: str = "gold_label",
):
    """
    Performs evaluation on evaluation documents in the DocumentStore. Returns a dict containing the following metrics:

          - "EM": Proportion of exact matches of predicted answers with their corresponding correct answers
          - "f1": Average overlap between predicted answers and their corresponding correct answers
          - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

    :param label_origin: the label for the correct answers 
    :param k_retriever: the number of answers to retrieve from the TitleQAPipeline 
    :param pipeline: The titleQAPipeline 
    :param document_store: DocumentStore containing the evaluation documents and the embeddings of the titles 
    :param label_index: Index/Table name where labeled questions are stored
    """

    # extract all questions for evaluation
    filters = {"origin": [label_origin]}
    # labels = document_store.get_all_labels(index=label_index, filters=filters)
    labels_agg = document_store.get_all_labels_aggregated(
        index=label_index, filters=filters
    )
    labels_agg = [label for label in labels_agg if label.question]

    questions = [label.question for label in labels_agg]
    predicted_answers_list = [
        pipeline.run(query=q, top_k_retriever=k_retriever) for q in questions
    ]
    assert len(questions) == len(predicted_answers_list), (
        f"Number of questions is not the same number of predicted" f"answers"
    )
    # quick renaming fix to match with haystack.eval.eval_counts_reader, this might be due to preprocessing
    # TODO : check this
    for predicted_answers in predicted_answers_list:
        for answer in predicted_answers["answers"]:
            answer["offset_start_in_doc"] = 0
            answer["offset_end_in_doc"] = len(answer["answer"])

    metric_counts = {}
    metric_counts["correct_no_answers_top1"] = 0
    metric_counts["correct_no_answers_topk"] = 0
    metric_counts["correct_readings_topk"] = 0
    metric_counts["exact_matches_topk"] = 0
    metric_counts["summed_f1_topk"] = 0
    metric_counts["correct_readings_top1"] = 0
    metric_counts["correct_readings_top1_has_answer"] = 0
    metric_counts["correct_readings_topk_has_answer"] = 0
    metric_counts["summed_f1_top1"] = 0
    metric_counts["summed_f1_top1_has_answer"] = 0
    metric_counts["exact_matches_top1"] = 0
    metric_counts["exact_matches_top1_has_answer"] = 0
    metric_counts["exact_matches_topk_has_answer"] = 0
    metric_counts["summed_f1_topk_has_answer"] = 0
    metric_counts["number_of_no_answer"] = 0
    for question, predicted_answers in zip(labels_agg, predicted_answers_list):
        metric_counts = eval_counts_reader(question, predicted_answers, metric_counts)
    metrics = calculate_reader_metrics(metric_counts, len(predicted_answers_list))
    metrics.update(metric_counts)
    return metrics
