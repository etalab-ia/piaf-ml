'''
Converts the SPF produced jsons (with prepare_spf_base) into a SQuAD type file + the 100 QR from the question fiches.
In short, it is based on the QR-fiches dataset prepared before (by Julien, 100 QR filtered_spf_qr_test.json file) and
adds to it the rest of the fiches from the JSON SPF fiches files.

   {
    version: "Version du dataset"
    data:[
            {
                title: "Fiche ID"
                paragraphs:[
                    {
                        context: "Fiche content"
                        qas:[
                            {
                                id: "Id du pair question-réponse"
                                question: "Question"
                                answers:[
                                    {
                                        "answer_start": "Position de la réponse"
                                        "text": "Réponse"
                                    }
                                ]
                                has_answer: True
                            }
                        ]
                    }
                ]
            }
    ]
}

Usage:
    spf2squad.py <json_fiches> <qr_dataset> <output_dataset> [options]

Arguments:
    <json_fiches>           JSON files that contain the spf fiches
    <qr_dataset>            SQuAD format JSON with the 100 QR dataset
    <output_dataset>        Merged dataset output path
    --optional_1 OPT1       Only convert those DOCs that are missing
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import hashlib
import json
import logging
from glob import glob
from pathlib import Path
from typing import List, Dict

from argopt import argopt
from tqdm import tqdm


def merge_SQuAD_datasets(qr_dataset: Path, spf_articles: Dict):
    """
    Merge both the SPF created dataaset in spf_spf_articles and the QR dataset qr_dataset into a single
    SQuAD dataset
    :param qr_dataset: Path of file containing the SPF SQuAD QR dataset
    :param spf_articles: Dict with the articles of the SPF fiches
    :return: Dict with the merged SQuAD dataset
    """
    if not qr_dataset.exists():
        raise Exception(f"File {qr_dataset} does not exist. Exiting...")
    with open(qr_dataset) as filo:
        qr_squad = json.load(filo)

    qr_data = qr_squad["data"]
    qr_articles = {t["title"]: t for t in qr_data}

    articles_list = []
    tqdm.write("Now we merge both dataset by iterating the created SPF fiches")
    for fiche_id, fiche_article in tqdm(spf_articles.items()):
        if fiche_id in qr_articles:
            article = qr_articles[fiche_id]
            article["title"] = spf_articles[fiche_id]["title"]  # TODO: Should we put the title as the fiche ID??
            for par in article["paragraphs"]:
                for qa in par["qas"]:
                    qa["is_impossible"] = False
        else:
            article = fiche_article
        articles_list.append(article)


    squad_dict = {
            "version": "1.0",
            "data": articles_list
        }



    return squad_dict


def run(doc_path: Path):
    """
    Generates an "article" structure following the SQuAD format.

    :param doc_path: Path of a fiche JSON file
    :return: A dict with the content of an SQuAD article
    """

    # 1. Get info from json file
    with open(doc_path) as filo:
        fiche_json = json.load(filo)
    if "arborescence" not in fiche_json or not fiche_json["arborescence"]:
        return {}
    fiche_text = fiche_json["text"]
    arborescence = fiche_json["arborescence"]

    fiche_title = arborescence["fiche"]
    fiche_id = arborescence["id"]

    # 2. Create article dict structure

    paragraph_dict = {
        "context": fiche_text,
        "qas": [
            {
                "id": hashlib.sha1(fiche_text.encode()).hexdigest()[:10],
                "question": "",
                "answers": [
                    {
                        "answer_start": -1,
                        "text": ""
                    }
                ],
                "is_impossible": True
            }
        ]
    }
    article_dict = {
        "title": fiche_title,
        "paragraphs": [paragraph_dict]
    }

    return {fiche_id: article_dict}


def main(doc_files_path: Path, qr_dataset_path: Path, output_dataset: Path):
    if not doc_files_path.exists():
        raise Exception(f"Path {doc_files_path} does not exist. Exiting...")

    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        doc_paths = [Path(p) for p in glob(doc_files_path.as_posix() + "/**/*.json", recursive=True)]
    if not doc_paths:
        raise Exception(f"Path {doc_paths} does not contain JSON files.")

    spf_articles = {}
    for doc_path in tqdm(doc_paths):
        tqdm.write(f"Converting file {doc_path}")
        spf_articles.update(run(doc_path))

    merged_dataset = merge_SQuAD_datasets(qr_dataset=qr_dataset_path, spf_articles=spf_articles)

    with open(output_dataset, "w") as filo:
        json.dump(merged_dataset, filo)


    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.json_fiches)
    qr_dataset = Path(parser.qr_dataset)
    output_dataset = Path(parser.output_dataset)
    main(doc_files_path=doc_files_path, qr_dataset_path=qr_dataset, output_dataset=output_dataset)
