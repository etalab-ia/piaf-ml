"""
Split PIAF dataset in train and test while keeping the following constraints:
 1. Do not mix contexts (Wikipedia paragraphs) in train and test.
 2. Keep only the questions with >3 answers in the test set

"""

import json
from typing import List

with open("./data/piaf_v1.1.json") as filo:
    piaf = json.load(filo)

piaf_test = {"version": "1.1", "data": []}
articles = piaf["data"]

for article in articles:
    test_article_dict = {"title": article["title"], "paragraphs": []}
    paragraphs: List = article["paragraphs"]
    temp_id = -1
    for idx, paragraph in enumerate(paragraphs):
        qas = paragraph["qas"]

        index_3_and_more = [len(qa["answers"]) for qa in qas]  # [1, 1, 3]

        if any([v >= 3 for v in index_3_and_more]):
            test_article_dict["paragraphs"].append(paragraph)
            temp_id = idx
    if temp_id >= 0:
        paragraphs.pop(temp_id)
        piaf_test["data"].append(test_article_dict)

piaf_test["version"] = "1.1"
with open("./data/piaf_test.json", "w") as test:
    json.dump(piaf_test, test)

with open("./data/piaf_train.json", "w") as train:
    json.dump(piaf, train)
