from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import fetch_archive_from_http
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.transformers import TransformersReader
from haystack.reader.farm import FARMReader
from haystack.finder import Finder
from farm.utils import initialize_device_settings

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

##############################################
# Settings
##############################################
LAUNCH_ELASTICSEARCH = True

eval_retriever_only = False
eval_reader_only = False
eval_both = True

##############################################
# Code
##############################################
device, n_gpu = initialize_device_settings(use_cuda=True)
# Start an Elasticsearch server
# You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
# your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.
if LAUNCH_ELASTICSEARCH:
    logging.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
    )
    if status.returncode:
        raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                        "then set LAUNCH_ELASTICSEARCH in the script to False.")
    time.sleep(30)

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                            create_index=False)
# Add evaluation data to Elasticsearch database
if LAUNCH_ELASTICSEARCH:
    document_store.add_eval_data("../../data/spf_qr_test.json")
else:
    logger.warning("Since we already have a running ES instance we should not index the same documents again."
                   "If you still want to do this call:"
                   "'document_store.add_eval_data('../../data/spf_qr_test.json')' manually ")

# Initialize Retriever
retriever = ElasticsearchRetriever(document_store=document_store)

# Initialize Reader
reader = TransformersReader(model="etalab-ia/camembert-base-squadFR-fquad-piaf",
                            tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf", use_gpu=-1)

# Initialize Finder which sticks together Reader and Retriever
finder = Finder(reader, retriever)

# Evaluate Retriever on its own
if eval_retriever_only:
    retriever_eval_results = retriever.eval()
    # Retriever Recall is the proportion of questions for which the correct document containing the answer is
    # among the correct documents
    print("Retriever Recall:", retriever_eval_results["recall"])
    # Retriever Mean Avg Precision rewards retrievers that give relevant documents a higher rank
    print("Retriever Mean Avg Precision:", retriever_eval_results["map"])

# Evaluate Reader on its own
if eval_reader_only:
    reader_eval_results = reader.eval(document_store=document_store, device=device)
    # Evaluation of Reader can also be done directly on a SQuAD-formatted file without passing the data to Elasticsearch
    # reader_eval_results = reader.eval_on_file("../data/nq", "nq_dev_subset_v2.json", device=device)

    # Reader Top-N-Accuracy is the proportion of predicted answers that match with their corresponding correct answer
    print("Reader Top-N-Accuracy:", reader_eval_results["top_n_accuracy"])
    # Reader Exact Match is the proportion of questions where the predicted answer is exactly the same as the correct answer
    print("Reader Exact Match:", reader_eval_results["EM"])
    # Reader F1-Score is the average overlap between the predicted answers and the correct answers
    print("Reader F1-Score:", reader_eval_results["f1"])

# Evaluate combination of Reader and Retriever through Finder
if eval_both:
    finder_eval_results = finder.eval(top_k_retriever=10, top_k_reader=10)
    finder.print_eval_results(finder_eval_results)
