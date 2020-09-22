"""
Transforms service public france fiches in XML format to txt. It tries to extract the essential content from the fiches

Usage:
    fiches_xml2txt.py <file_path> <output_path> [options]

Arguments:
    <file_path>             A path of a single XML fiche or a folder with multiple fiches XML
    <output_path>           A path where to store the extracted info
    --cores=<n> CORES       Number of cores to use [default: 1:int]
"""

from xml.etree.ElementTree import Element
from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm

import unicodedata
import re
import json


def extract_arbo(doc_path):
    """
    Extracts arbo from a single fiche
    """
    tree = ET.parse(doc_path)
    root = tree.getroot()
    fil_ariane = list(root.iter("FilDAriane"))[0]
    arbo = [x.text for x in fil_ariane]
    return arbo


def main(doc_files_path, output_path, n_jobs):

    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path.as_posix() + "/**/F*.xml", recursive=True)
        doc_paths += glob(doc_files_path.as_posix() + "/**/N*.xml", recursive=True)
        doc_paths = [Path(p) for p in doc_paths]
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    path = output_path / 'arborescence.json'
    arborescence = {}

    for doc_path in tqdm(doc_paths):

        arbo = extract_arbo(doc_path)

        for i in range(len(arbo)):
            if i == 0:
                categorie = arbo[i]
                if categorie not in arborescence.keys():
                    arborescence[categorie] = {}
            elif i == 1:
                categorie, theme = arbo[i - 1], arbo[i]
                if theme not in arborescence[categorie].keys():
                    arborescence[categorie][theme] = {}
            elif i == 2:
                categorie, theme, dossier = arbo[i - 2], arbo[i - 1], arbo[i]
                if dossier not in arborescence[categorie][theme].keys():
                    arborescence[categorie][theme][dossier] = []
            elif i == 3:
                categorie, theme, dossier, sous_dossier = arbo[i - 3], arbo[i - 2], arbo[i - 1], arbo[i]
                if sous_dossier not in arborescence[categorie][theme][dossier]:
                    arborescence[categorie][theme][dossier].append(sous_dossier)
            else:
                tddm.print("Fiche arborescence is deeper than 4!")

    with open(path.as_posix(), "w", encoding='utf-8') as out_file:
        json.dump(arborescence, out_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.file_path)
    output_path = Path(parser.output_path)
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, output_path=output_path, n_jobs=n_jobs)