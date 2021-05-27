"""
Script to convert a DPR Question-Contexts format JSON file to a SQuAD Json Format

Usage: dpr2squad.py <dpr_file_path> <squad_output_path> [options]

Arguments:
    <dpr_file_path>   DPR file path
    <squad_output_path>   SQuAD outpput file path
"""
import hashlib
import json
import random
from pathlib import Path

from argopt import argopt
from tqdm import tqdm

random.seed(42)

"""
Transforms this: [ { "question": "....", "answers": ["...", "...", "..."], "positive_ctxs": [{ "title": "...", "text":
"...." }], "negative_ctxs": ["..."], "hard_negative_ctxs": ["..."] }, ... ] to this: { data:[ { "title": "paragraphs":[{
"context": "qas":[ { "answers":[ { "answer_start": "text": ], "id":, "question":, "is_impossible": False } } ] ] } ] }
"""


def create_squad_eval_dataset(dpr_file_path: Path):
    with open(dpr_file_path.as_posix()) as o:
        dpr_file = json.load(o)
    squad_data = {"data": []}
    for question in tqdm(dpr_file):
        doc_dict = {}
        paragraph_dict = {
            "context": question["positive_ctxs"][0]["text"],
            "qas": [
                {
                    "answers": [{"answer_start": 0, "text": question["answers"][0]}],
                    "question": question["question"],
                    "is_impossible": False,
                    "id": hashlib.md5(
                        str(question["question"]).encode("utf-8")
                    ).hexdigest()[:8],
                }
            ],
        }
        doc_dict["title"] = question["positive_ctxs"][0]["title"]
        doc_dict["paragraphs"] = [paragraph_dict]
        squad_data["data"].append(doc_dict)
    return squad_data


def main(dpr_file_path: Path, squad_output_path: Path):
    tqdm.write(f"Using DPR file {dpr_file_path}")
    squad_dataset = create_squad_eval_dataset(dpr_file_path)
    with open(squad_output_path.as_posix(), "w") as o:
        json.dump(squad_dataset, o, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argopt(__doc__).parse_args()
    dpr_file_path = Path(parser.dpr_file_path)
    squad_output_path = Path(parser.squad_output_path)
    main(dpr_file_path=dpr_file_path, squad_output_path=squad_output_path)
