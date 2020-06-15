'''
Transforms service public france fiches in XML format to txt. It tries to extract the essential content from the fiches

Usage:
    fiches_xml2txt.py <some_path> [options]

Arguments:
    <file_path>             A path of a single XML fiche or a folder with multiple fiches XML
    --optional_1 OPT1       Only convert those DOCs that are missing
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import logging
import os
from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm


def run(doc_path):
    tree = ET.parse(doc_path)
    root = tree.getroot()

    text_list = []
    for neighbor in root.iter('Paragraphe'):
        text = " ".join(list(neighbor.itertext())).strip("\n")
        text_list.append(text)

    for text in text_list:
        print(text + "\n")

    # with doc_path

    return 1


def main(doc_files_path: Path, optional_1, n_jobs: int):
    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path.as_posix() + "/**/*.xml", recursive=True)
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if n_jobs < 2:
        job_output = []
    for doc_path in tqdm(doc_paths):
        tqdm.write(f"Converting file {doc_path}")
        job_output.append(run(doc_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(run)(doc_path) for doc_path in tqdm(doc_paths))

    logging.info(
        f"{sum(job_output)} DOC files were converted to TXT. {len(job_output) - sum(job_output)} files "
        f"had some error.")

    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.file_path)
    optional_1 = parser.optional_1
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, optional_1=optional_1, n_jobs=n_jobs)
