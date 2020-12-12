from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.transformers import TransformersReader
from haystack.reader.farm import FARMReader
from haystack.finder import Finder
from farm.utils import initialize_device_settings

import logging
import subprocess
import time
import json

from src.evaluation.retriever_25k_eval import launch_ES

logger = logging.getLogger(__name__)

##############################################
# Settings
##############################################
eval_retriever_only = False
eval_reader_only = False
eval_both = True

# Create our own indexes names as default ones are sometimes buggy
eval_qr_filename = "./data/filtered_spf_qr_test.json"
doc_index = "piaf_eval_docs"
label_index = "piaf_eval_labels"

##############################################
# Code
##############################################
device, n_gpu = initialize_device_settings(use_cuda=False)
# Start an Elasticsearch server
# You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
# your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.

launch_ES()

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                            create_index=False)
document_store.delete_all_documents(index=doc_index)
document_store.delete_all_documents(index=label_index)
# Add evaluation data to Elasticsearch database
# TODO: change with later Haystack version with delete_all_documents(index=...), see Tutorial 5 Evaluation on github
document_store.add_eval_data(filename=eval_qr_filename, doc_index=doc_index, label_index=label_index)

# Checking data integrity
with open("./data/filtered_spf_qr_test.json") as file:
    n_docs = len(json.load(file, encoding='utf-8')['data'])
n_eval_data = document_store.get_document_count(index=doc_index)
print(f"Number of documents in eval data is {n_eval_data}, should be {n_docs}")

# Initialize Retriever
retriever = ElasticsearchRetriever(document_store=document_store)

# Initialize Reader
reader = FARMReader("etalab-ia/camembert-base-squadFR-fquad-piaf", use_gpu=False, top_k_per_candidate=4)

# Initialize Finder which sticks together Reader and Retriever
finder = Finder(reader, retriever)

# Evaluate Retriever on its own
if eval_retriever_only:
    for top_k in [10, 5]:
        # TODO: check if next line works with newer Haystack version (should be OK)
        retriever_eval_results = retriever.eval(top_k=top_k, doc_index=doc_index, label_index=label_index)
        # Retriever Recall is the proportion of questions for which the correct document containing the answer is
        # among the correct documents
        print(f"Retriever Recall with top_k={top_k}: {retriever_eval_results['recall']}")
        # Retriever Mean Avg Precision rewards retrievers that give relevant documents a higher rank
        print(f"Retriever Mean Avg Precision with top_k={top_k}: {retriever_eval_results['map']}")

# Evaluate Reader on its own
if eval_reader_only:
    # TODO: check if next line works with newer Haystack version
    reader_eval_results = reader.eval(document_store=document_store, device=device,
                                      doc_index=doc_index, label_index=label_index)
    # Evaluation of Reader can also be done directly on a SQuAD-formatted file without passing the data to Elasticsearch
    # reader_eval_results = reader.eval_on_file("../data/nq", "nq_dev_subset_v2.json", device=device)

    # Reader Top-N-Accuracy is the proportion of predicted answers that match with their corresponding correct answer
    print("Reader Top-N-Accuracy:", reader_eval_results["top_n_accuracy"])
    # Reader Exact Match is the proportion of questions where the predicted answer is exactly the same
    # as the correct answer
    print("Reader Exact Match:", reader_eval_results["EM"])
    # Reader F1-Score is the average overlap between the predicted answers and the correct answers
    print("Reader F1-Score:", reader_eval_results["f1"])

# Evaluate combination of Reader and Retriever through Finder
if eval_both:
    finder_eval_results = finder.eval(top_k_retriever=10, top_k_reader=5, doc_index=doc_index, label_index=label_index)
    finder.print_eval_results(finder_eval_results)
