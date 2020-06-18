'''
Transforms service public france fiches in XML format to txt. It tries to extract the essential content from the fiches

Usage:
    fiches_xml2txt.py <file_path> [options]

Arguments:
    <file_path>             A path of a single XML fiche or a folder with multiple fiches XML
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import logging

from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm


def run(doc_path: Path):
    try:
        tqdm.write(f"Extracting info from {doc_path}")
        tree = ET.parse(doc_path)
        root = tree.getroot()

        text_list = []
        for neighbor in root.iter('Paragraphe'):
            text = " ".join(list(neighbor.itertext())).strip("\n")
            text_list.append(text)

        text_path = doc_path[:-4] + ".txt"
        with open(text_path, "w") as filo:
            for text in text_list:
                filo.write(text + "\n\n")
                # print(text + "\n")

        return 1
    except Exception as e:
        tqdm.write(f"Could not treat file {doc_path}. Error: {str(e)}")
        return 0


def main(doc_files_path: Path, n_jobs: int):
    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path.as_posix() + "/**/F*.xml", recursive=True)
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if n_jobs < 2:
        job_output = []
        for doc_path in tqdm(doc_paths):
            tqdm.write(f"Converting file {doc_path}")
            job_output.append(run(doc_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(run)(doc_path) for doc_path in tqdm(doc_paths))

    tqdm.write(
        f"{sum(job_output)} DOC files were converted to TXT. {len(job_output) - sum(job_output)} files "
        f"had some error.")

    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.file_path)
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, n_jobs=n_jobs)
