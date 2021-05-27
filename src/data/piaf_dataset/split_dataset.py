"""
Split PIAF dataset in train and test while keeping the following constraints:

 1. Do not mix contexts (Wikipedia paragraphs) in train and test.
 2. Keep only the questions with >3 answers in the test set

"""

import json
from typing import List

VERSION = "1.1"
with open("./data/piaf_v1.1.json") as filo:
    piaf_train = json.load(filo)

piaf_test = {"version": VERSION, "data": []}
articles = piaf_train["data"]

for article in articles:
    test_article_dict = {"title": article["title"], "paragraphs": []}
    paragraphs: List = article["paragraphs"]
    temp_id = -1
    for idx, context in enumerate(paragraphs):
        qas = context["qas"]

        index_3_and_more = [
            (idx, len(qa["answers"])) for idx, qa in enumerate(qas)
        ]  # [1, 1, 3]

        if any([v[1] >= 3 for v in index_3_and_more]):
            # if we have a question with more than 3 answers we add it to the test dataset
            # but first we should remove the questions that do not have >=3 questions
            test_context = dict(context)
            test_context.update(
                {"qas": [qas[v[0]] for v in index_3_and_more if v[1] >= 3]}
            )
            test_article_dict["paragraphs"].append(test_context)
            temp_id = idx
    if temp_id >= 0:
        # if we had a question with >=3 answers, we remove the context from the train dataset``
        paragraphs.pop(temp_id)
        piaf_test["data"].append(test_article_dict)

piaf_test["version"] = VERSION
with open("./data/piaf_test.json", "w") as test:
    json.dump(piaf_test, test)

with open("./data/piaf_train.json", "w") as train:
    json.dump(piaf_train, train)
