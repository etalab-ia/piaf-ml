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
    audience = list(root.iter("Audience"))[0].text
    titre = list(root.findall(".//dc:title", namespaces={'dc': 'http://purl.org/dc/elements/1.1/'}))[0].text

    # Le theme n'est vide que quand il s'agit d'une fiche "Comment faire si..."
    try:
        theme = list(list(root.iter("Theme"))[0])[0].text
    except IndexError:
        theme = ''

    # Cinq cas différents (cas de base + 4 types de fiches spéciaux), on les traite différemment
    try:
        dossier = list(list(root.iter("DossierPere"))[0])[0].text
    except IndexError:
        dossier = list(root.findall(".//dc:type", namespaces={'dc': 'http://purl.org/dc/elements/1.1/'}))[0].text
        if dossier in ['Question-réponse', 'Comment faire si...']:
            theme = ''
        elif dossier == 'Dossier':
            pass
        elif dossier == 'Thème':
            return None

    # Sous-dossiers pas toujours présents
    try:
        sous_dossier = list(root.iter("SousDossierPere"))[0].text
    except IndexError:
        sous_dossier = ''

    return {'audience': audience, 'theme': theme, 'dossier': dossier, 'sous_dossier': sous_dossier, 'titre': titre}


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
        if arbo is not None:
            audience, theme = arbo['audience'], arbo['theme']
            dossier, sous_dossier, titre = arbo['dossier'], arbo['sous_dossier'], arbo['titre']
            if audience not in arborescence.keys():
                arborescence[audience] = {}
            if theme not in arborescence[audience].keys():
                arborescence[audience][theme] = {}
            if dossier not in arborescence[audience][theme].keys():
                arborescence[audience][theme][dossier] = {}
            if sous_dossier not in arborescence[audience][theme][dossier].keys():
                arborescence[audience][theme][dossier][sous_dossier] = []
            if titre not in arborescence[audience][theme][dossier][sous_dossier]:
                arborescence[audience][theme][dossier][sous_dossier].append(titre)

    with open(path.as_posix(), "w", encoding='utf-8') as out_file:
        json.dump(arborescence, out_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.file_path)
    output_path = Path(parser.output_path)
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, output_path=output_path, n_jobs=n_jobs)