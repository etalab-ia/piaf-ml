import logging

from tqdm import tqdm

from haystack.document_store.base import BaseDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.pipeline import BaseStandardPipeline
from haystack.eval import eval_counts_reader, calculate_reader_metrics, calculate_average_precision_and_reciprocal_rank

logger = logging.getLogger(__name__)


def eval_retriever(
    document_store: BaseDocumentStore,
    pipeline: BaseStandardPipeline,
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

    correct_retrievals = 0
    summed_avg_precision = 0.0
    summed_reciprocal_rank = 0.0

    # Collect questions and corresponding answers/document_ids in a dict
    question_label_dict = {}
    for label in labels:
        deduplicated_doc_ids = list(set([str(x) for x in label.multiple_document_ids]))
        question_label_dict[label.question] = deduplicated_doc_ids


    predictions = []
    """
    confusion_matrix_documents = 0
    for i in range(top_k_reader+1):
        pred_docs = []
        for answer, true_docs_ids in zip(found_answers, true_docs_ids_list):
            pred_docs_id = answer['answers'][i]['document_id']
            print(pred_docs_id)
            print(true_docs_ids)
            if pred_docs_id in true_docs_ids:
                print('OK')
                pred_docs.append(1)
            else:
                pred_docs.append(0)
        confusion_matrix_documents += confusion_matrix([1 for i in range(len(true_answers_list))], pred_docs)
    confusion_matrix_documents = confusion_matrix_documents / top_k_reader"""

    # Option 1: Open-domain evaluation by checking if the answer string is in the retrieved docs
    logger.info("Performing eval queries...")

    for question, gold_ids in tqdm(question_label_dict.items()):
        number_relevant_docs = len(gold_ids)
        retrieved_docs = pipeline.run(query=question, top_k_retriever=top_k)
        if return_preds:
            predictions.append({"question": question, "retrieved_docs": retrieved_docs['documents']})
        # check if correct doc in retrieved docs
        found_relevant_doc = False
        relevant_docs_found = 0
        current_avg_precision = 0.0
        for doc_idx, doc in enumerate(retrieved_docs['documents']):
            if str(doc.id) in gold_ids:
                relevant_docs_found += 1
                print('relevant_docs_found:' + str(relevant_docs_found))
                if not found_relevant_doc:
                    print('correct_retrievals:' + str(correct_retrievals))
                    summed_reciprocal_rank += 1 / (doc_idx + 1)
                current_avg_precision += relevant_docs_found / (doc_idx + 1)
                found_relevant_doc = True
                if relevant_docs_found == number_relevant_docs:
                    break
        if found_relevant_doc:
            all_relevant_docs = len(set(gold_ids))
            summed_avg_precision += current_avg_precision / all_relevant_docs
    # Metrics
    number_of_questions = len(question_label_dict)
    recall = correct_retrievals / number_of_questions
    mean_reciprocal_rank = summed_reciprocal_rank / number_of_questions
    mean_avg_precision = summed_avg_precision / number_of_questions

    logger.info((f"For {correct_retrievals} out of {number_of_questions} questions ({recall:.2%}), the answer was in"
                 f" the top-{top_k} candidate passages selected by the retriever."))

    metrics =  {
        "recall": recall,
        "map": mean_avg_precision,
        "mrr": mean_reciprocal_rank,
        #"retrieve_time": retriever.retrieve_time,
        "n_questions": number_of_questions,
        "top_k": top_k
    }

    if return_preds:
        return {"metrics": metrics, "predictions": predictions}
    else:
        return metrics



def eval_retriever_reader(
        document_store: BaseDocumentStore,
        pipeline: BaseStandardPipeline,
        top_k_reader: int,
        top_k_retriever: int,
        label_index: str = "label",
        doc_index: str = "document",
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
    #labels = document_store.get_all_labels(index=label_index, filters=filters)
    labels_agg = document_store.get_all_labels_aggregated(index=label_index, filters=filters)


    questions = [label.question for label in labels_agg]
    predicted_answers_list = [pipeline.run(query=q, top_k_retriever=top_k_retriever) for q in questions]

    metric_counts={}
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
    for question, predicted_answers in zip(labels_agg,predicted_answers_list):
        metric_counts = eval_counts_reader(question,predicted_answers,metric_counts)
    metrics = calculate_reader_metrics(metric_counts,len(predicted_answers))

    return metrics










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