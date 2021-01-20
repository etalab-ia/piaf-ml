'''
This script creates a JSON file with fiches from service-public.fr that contain a question as title. It stores
the question (the fiche title) and the text content of the fiche in a dict structure.
The produced JSON file is known as the Questions Fiches dataset.

Usage:
    create_questions_dataset.py <file_path> [options]

Arguments:
    <file_path>                     A required path parameter
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import json
import logging
import os
import subprocess
from glob import glob
from pathlib import Path

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm
import xml.etree.ElementTree as ET


def run(doc_path):
    try:
        file_name = Path(doc_path).stem
        tqdm.write(f"Extracting info from {doc_path}")
        tree = ET.parse(doc_path)
        root = tree.getroot()
        fiche_title = list(list(root.iter("Publication"))[0])[0].text
        intro = list(root.iter("Introduction"))
        if "?" not in fiche_title[-5:] or not intro:
            tqdm.write(f"{doc_path} is not a question fiche")
            return 0

        qfiche_text_list = []
        for cas_paragraph in root.iter("Paragraphe"):
            qfiche_text_list.append(list(cas_paragraph.itertext()))
        qfiche_text_list = [" ".join(t) for t in qfiche_text_list]
        qfiche_text = "\n".join(qfiche_text_list)

        if qfiche_text:
            first_sentence = qfiche_text
            print(fiche_title)
            print(f"\t{first_sentence}")
            return file_name, fiche_title, first_sentence

    except:
        tqdm.write(f"Could not treat {doc_path}")
        return 0


def main(doc_files_path: Path, n_jobs: int):
    if not os.path.isdir(doc_files_path) and os.path.isfile(doc_files_path):
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path + "/**/F*.xml", recursive=True)
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if n_jobs < 2:
        job_output = []
        for doc_path in tqdm(doc_paths):
            tqdm.write(f"Converting file {doc_path}")
            job_output.append(run(doc_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(run)(doc_path) for doc_path in tqdm(doc_paths))

    clean_job_output = [j for j in job_output if j]
    with open("./data/questions_spf.json", "w") as outo:
        json.dump(clean_job_output, outo, indent=4, ensure_ascii=False)

    tqdm.write(f"There are {len(clean_job_output)} fiches type question")
    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = parser.file_path
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, n_jobs=n_jobs)
