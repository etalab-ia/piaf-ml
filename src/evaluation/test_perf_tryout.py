import json
import logging
import subprocess
import time
from random import sample, seed
seed(42)
from tqdm import tqdm

from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.retriever.sparse import ElasticsearchRetriever

logger = logging.getLogger(__name__)

LAUNCH_ELASTICSEARCH = True


if LAUNCH_ELASTICSEARCH:
    logging.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
    )
    if status.returncode:
        raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                        "then set LAUNCH_ELASTICSEARCH in the script to False.")
    time.sleep(15)

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

doc_dir = "/data/service-public-france/extracted/"

dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the docs to our DB.
if LAUNCH_ELASTICSEARCH:
    document_store.write_documents(dicts)
else:
    logger.warning("Since we already have a running ES instance we should not index the same documents again. \n"
                   "If you still want to do this call: document_store.write_documents(dicts) manually ")


retriever = ElasticsearchRetriever(document_store=document_store)

reader = TransformersReader(model="/models/piaf/6_Squad-FR_train_Fquad-train_PIAF",
                            tokenizer="/models/piaf/6_Squad-FR_train_Fquad-train_PIAF", use_gpu=-1)
# reader = TransformersReader(model="fmikaelian/camembert-base-fquad",
#                             tokenizer="fmikaelian/camembert-base-fquad", use_gpu=-1)
finder = Finder(reader, retriever)

fiches_qas = json.load(open("/home/pavel/code/piaf-ml/data/questions_spf.json"))
sample_qas = sample(fiches_qas, 25)

match_fiche = 0
results = []
for fiche_name_true, question, answer in tqdm(sample_qas):
    prediction = finder.get_answers(question, top_k_retriever=5, top_k_reader=1)
    first_answer = prediction["answers"][0]
    fiche_name_pred = first_answer["meta"]["name"]
    prob = first_answer["probability"]
    if fiche_name_true in fiche_name_pred:
        match_fiche += 1
    results.append({"question": question, "true_answer": answer,
                    "pred_answer": first_answer["answer"],
                    "pred_fiche": fiche_name_pred,
                    "true_fiche": fiche_name_true,
                    "probability": prob
                    })

tqdm.write(f"Tried {len(sample_qas)} questions and got {len(results)}.")
tqdm.write(f"The retriever correctly found {match_fiche} fiches, so {match_fiche/len(sample_qas)}")

# print_answers(prediction, details="all")
json.dump(results, open("./data/evaluation_haystack.json", "w"), indent=4, ensure_ascii=False)