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
import subprocess
from glob import glob
from pathlib import Path

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm


def run(doc_path):
    return 1


def main(doc_files_path: Path, optional_1, n_jobs: int):
    doc_paths = []
    if not os.path.isdir(doc_files_path) and os.path.isfile(doc_files_path):
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path + "/**/*.doc", recursive=True)
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
    doc_files_path = parser.file_path
    optional_1 = parser.optional_1
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, optional_1=optional_1, n_jobs=n_jobs)
