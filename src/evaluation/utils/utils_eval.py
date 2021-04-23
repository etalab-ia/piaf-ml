import json
import logging
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from haystack.document_store.base import BaseDocumentStore
from haystack.eval import eval_counts_reader, calculate_reader_metrics
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
) -> dict:
    # Extract all questions for evaluation
    filters = {"origin": [label_origin]}

    labels = document_store.get_all_labels_aggregated(index=label_index, filters=filters)

    """
    true_answers_list = [label.multiple_answers for label in labels_agg]
    true_docs_ids_list = [list(set([str(x) for x in label.multiple_document_ids])) for label in labels_agg] #this is to deduplicate doc_ids
    questions = [label.question for label in labels_agg]
    found_answers = [pipeline.run(query=q, top_k_retriever=top_k_retriever) for q in questions]"""

    # Collect questions and corresponding answers/document_ids in a dict
    question_label_dict_list = []
    for label in labels:
        if not label.question:
            continue
        deduplicated_doc_ids = list(set([str(x) for x in label.multiple_document_ids]))
        question_label_dict = {'query': label.question, 'gold_ids': deduplicated_doc_ids}
        question_label_dict_list.append(question_label_dict)

    retrieved_docs_list = [pipeline.run(query=question["query"], top_k_retriever=top_k, index=doc_index) for question in
                           question_label_dict_list]

    metrics = get_retriever_metrics(retrieved_docs_list, question_label_dict_list)

    logger.info((
        f"For {metrics['correct_retrievals']} out of {metrics['number_of_questions']} questions ({metrics['recall']:.2%}), the answer was in"
        f" the top-{top_k} candidate passages selected by the retriever."))

    return metrics


def get_retriever_metrics(retrieved_docs_list, question_label_dict_list):
    correct_retrievals = 0
    summed_avg_precision = 0.0
    summed_reciprocal_rank = 0.0

    for question, retrieved_docs in tqdm(zip(question_label_dict_list, retrieved_docs_list)):
        gold_ids = question['gold_ids']
        number_relevant_docs = len(gold_ids)
        # check if correct doc in retrieved docs
        found_relevant_doc = False
        relevant_docs_found = 0
        current_avg_precision = 0.0
        for doc_idx, doc in enumerate(retrieved_docs['documents']):
            if str(doc.id) in gold_ids:
                relevant_docs_found += 1
                if not found_relevant_doc:
                    correct_retrievals += 1
                    summed_reciprocal_rank += 1 / (doc_idx + 1)
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
    mean_reciprocal_rank = summed_reciprocal_rank / number_of_questions
    mean_avg_precision = summed_avg_precision / number_of_questions

    metrics = {
        "recall": recall,
        "map": mean_avg_precision,
        "mrr": mean_reciprocal_rank,
        "correct_retrievals": correct_retrievals,
        "number_of_questions": number_of_questions
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
    Performs evaluation on evaluation documents in the DocumentStore.
    Returns a dict containing the following metrics:
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
    labels_agg = document_store.get_all_labels_aggregated(index=label_index, filters=filters)
    labels_agg = [label for label in labels_agg if label.question]

    questions = [label.question for label in labels_agg]
    predicted_answers_list = []
    for q in questions:
        answer = pipeline.run(query=q, top_k_retriever=k_retriever, top_k_reader=k_reader_total)
        predicted_answers_list.append(answer)

    assert len(questions) == len(predicted_answers_list), f"Number of questions is not the same number of predicted" \
                                                          f"answers"
    # quick renaming fix to match with haystack.eval.eval_counts_reader, this might be due to preprocessing
    # TODO : check this
    for predicted_answers in predicted_answers_list:
        for answer in predicted_answers['answers']:
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


def full_eval_retriever_reader(
        document_store: BaseDocumentStore,
        pipeline: Pipeline,
        k_retriever: int,
        k_reader_total: int,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
):
    filters = {"origin": [label_origin]}
    labels = document_store.get_all_labels_aggregated(index=label_index, filters=filters)

    question_label_dict_list = []
    for label in labels:
        if not label.question:
            continue
        deduplicated_doc_ids = list(set([str(x) for x in label.multiple_document_ids]))
        question_label_dict = {'query': label.question, 'gold_ids': deduplicated_doc_ids}
        question_label_dict_list.append(question_label_dict)


    predicted_answers_list = [pipeline.run(query=question["query"], top_k_retriever=k_retriever) for question in
                           question_label_dict_list]
   

    retriever_metrics = get_retriever_metrics(predicted_answers_list, question_label_dict_list)

    assert len(question_label_dict_list) == len(predicted_answers_list), f"Number of questions is not the same number of predicted" \
                                                          f"answers"

    for predicted_answers in predicted_answers_list:
        for answer in predicted_answers['answers']:
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
    for question, predicted_answers in zip(labels, predicted_answers_list):
        metric_counts = eval_counts_reader(question, predicted_answers, metric_counts)
    retriever_reader_metrics = calculate_reader_metrics(metric_counts, len(predicted_answers_list))
    retriever_reader_metrics.update(metric_counts)
    retriever_reader_metrics.update(retriever_metrics)

    return retriever_reader_metrics


"""
    # Aggregate all answer labels per question
    aggregated_per_doc = defaultdict(list)
    for label in labels:
        if not label.document_id:
            logger.error(f"Label does not contain a document_id")
            continue
        aggregated_per_doc[label.document_id].append(label)

    # Create squad style dicts
    d: Dict[str, Any] = {}
    all_doc_ids = [x.id for x in document_store.get_all_documents(doc_index)]
    for doc_id in all_doc_ids:
        doc = document_store.get_document_by_id(doc_id, index=doc_index)
        if not doc:
            logger.error(f"Document with the ID '{doc_id}' is not present in the document store.")
            continue
        d[str(doc_id)] = {
            "context": doc.text
        }
        # get all questions / answers
        aggregated_per_question: Dict[str, Any] = defaultdict(list)
        if doc_id in aggregated_per_doc:
            for label in aggregated_per_doc[doc_id]:
                # add to existing answers
                if label.question in aggregated_per_question.keys():
                    if label.offset_start_in_doc == 0 and label.answer == "":
                        continue
                    else:
                        # Hack to fix problem where duplicate questions are merged by doc_store processing creating a QA example with 8 annotations > 6 annotation max
                        if len(aggregated_per_question[label.question]["answers"]) >= 6:
                            continue
                        aggregated_per_question[label.question]["answers"].append({
                                    "text": label.answer,
                                    "answer_start": label.offset_start_in_doc})
                        aggregated_per_question[label.question]["is_impossible"] = False
                # create new one
                else:
                    # We don't need to create an answer dict if is_impossible / no_answer
                    if label.offset_start_in_doc == 0 and label.answer == "":
                        aggregated_per_question[label.question] = {
                            "id": str(hash(str(doc_id) + label.question)),
                            "question": label.question,
                            "answers": [],
                            "is_impossible": True
                        }
                    else:
                        aggregated_per_question[label.question] = {
                            "id": str(hash(str(doc_id)+label.question)),
                            "question": label.question,
                            "answers": [{
                                    "text": label.answer,
                                    "answer_start": label.offset_start_in_doc}],
                            "is_impossible": False
                        }

        # Get rid of the question key again (after we aggregated we don't need it anymore)
        d[str(doc_id)]["qas"] = [v for v in aggregated_per_question.values()]

    results = {
        "EM": eval_results[0]["EM"],
        "f1": eval_results[0]["f1"],
        "top_n_accuracy": eval_results[0]["top_n_accuracy"],
        "top_n": self.inferencer.model.prediction_heads[0].n_best,
        "reader_time": reader_time,
        "seconds_per_query": reader_time / n_queries
    }
    return results"""


def eval_titleQA_pipeline (
        document_store: BaseDocumentStore,
        pipeline: Pipeline,
        k_retriever: int,
        label_index: str = "label",
        label_origin: str = "gold_label",
):
    """
    Performs evaluation on evaluation documents in the DocumentStore.
    Returns a dict containing the following metrics:
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
    labels_agg = document_store.get_all_labels_aggregated(index=label_index, filters=filters)
    labels_agg = [label for label in labels_agg if label.question]

    questions = [label.question for label in labels_agg]
    predicted_answers_list = [pipeline.run(query=q, top_k_retriever=k_retriever) for q in questions]
    assert len(questions) == len(predicted_answers_list), f"Number of questions is not the same number of predicted" \
                                                          f"answers"
    # quick renaming fix to match with haystack.eval.eval_counts_reader, this might be due to preprocessing
    # TODO : check this
    for predicted_answers in predicted_answers_list:
        for answer in predicted_answers['answers']:
            answer["offset_start_in_doc"] = 0
            answer["offset_end_in_doc"] = len(answer['answer'])

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
