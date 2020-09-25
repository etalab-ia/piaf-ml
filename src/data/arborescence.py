"""
Transforms service public france fiches in XML format to txt. It tries to extract the essential content from the fiches

Usage:
    fiches_xml2txt.py <file_path> <output_path> [options]

Arguments:
    <file_path>             A path of a single XML fiche or a folder with multiple fiches XML
    <output_path>           A path where to store the extracted info
    --cores=<n> CORES       Number of cores to use [default: 1:int]
"""


from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET

from argopt import argopt
from tqdm import tqdm

import os
import json


def extract_arbo(doc_path):
    """
    Extracts arbo from a single fiche
    """

    tree = ET.parse(doc_path)
    root = tree.getroot()
    audience = list(root.iter("Audience"))[0].text
    titre = list(root.findall(".//dc:title", namespaces={'dc': 'http://purl.org/dc/elements/1.1/'}))[0].text
    fiche = list(root.findall(".//dc:identifier", namespaces={'dc': 'http://purl.org/dc/elements/1.1/'}))[0].text

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

    return {'audience': audience, 'theme': theme, 'dossier': dossier, 'sous_dossier': sous_dossier, 'titre': titre, 'fiche': fiche}

def concat_ID_and_name (id, name):
    return id + '//' + name

def extra_arbo_dossier(doc_path):
    """
    Extracts arbo from a single fiche N in order to get the part theme / sous_theme / dosser
    """

    tree = ET.parse(doc_path)
    root = tree.getroot()
    #for sous_theme in list(root.iter("SousTheme")):

    try:
        type = root.attrib['type']
    except:
        print("not correct file -->" + doc_path)
        return {}

    def iter_fiches (study_root):
        fiches = []
        for fiche in study_root.iter('Fiche'):
            fiche_id = fiche.attrib['ID']
            fiche_name = fiche.text
            fiche_txt = concat_ID_and_name(fiche_id, fiche_name)
            fiches.append(fiche_txt)
        return fiches

    if type == 'Dossier':
        theme = root.find('Theme').find('Titre').text
        theme_id = root.find('Theme').attrib['ID']
        theme_txt = concat_ID_and_name(theme_id, theme)
        try:
            sous_theme = root.find('SousThemePere').text #there might be no sous-theme
        except:
            sous_theme = ''
        dossier = list(root.findall(".//dc:title", namespaces={'dc': 'http://purl.org/dc/elements/1.1/'}))[0].text
        dossier_id = root.attrib['ID']
        dossier_txt = concat_ID_and_name(dossier_id, dossier)
        sous_dossier_dict = {}
        if root.find('SousDossier') != None: #case when there is one or several sous_dossier
            for ss_dossier in root.iter('SousDossier'):
                sous_dossier = ss_dossier.find('Titre').text
                list_fiches = iter_fiches(ss_dossier)
                sous_dossier_dict[sous_dossier] = list_fiches
        else: #case when there is no sous_dossier
            list_fiches = iter_fiches(root)
            sous_dossier_dict[""] = list_fiches
        dossier_dict = {dossier_txt: sous_dossier_dict}
        sous_theme_dict = {sous_theme: dossier_dict}
        theme_dict = {theme_txt: sous_theme_dict}
        return theme_dict
    else:
        return {}



def main(doc_files_path, output_path, n_jobs):

    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        # doc_paths = glob(doc_files_path.as_posix() + "/**/F*.xml", recursive=True)
        # go through the N files that contains the arbo
        doc_paths = glob(doc_files_path.as_posix() + "/**/N*.xml", recursive=True)
        doc_paths = [Path(p) for p in doc_paths]
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    path = output_path / 'arborescence.json'
    arborescence = {}

    for doc_path in tqdm(doc_paths):
        arbo = extra_arbo_dossier(doc_path)
        if len(arbo) != 0:
            theme = list(arbo.keys())[0]
            if theme not in arborescence.keys():
                arborescence[theme] = arbo[theme]
            else:
                sous_theme = list(arbo[theme].keys())[0]
                if sous_theme not in arborescence[theme].keys():
                    arborescence[theme][sous_theme]=arbo[theme][sous_theme]
                else:
                    dossier = list(arbo[theme][sous_theme].keys())[0]
                    if dossier not in arborescence[theme][sous_theme].keys():
                        arborescence[theme][sous_theme][dossier] = arbo[theme][sous_theme][dossier]
                    else:
                        print('This should not occur')

    if not output_path.exists():
        os.makedirs(output_path)

    with open(path.as_posix(), "w", encoding='utf-8') as out_file:
        json.dump(arborescence, out_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.file_path)
    output_path = Path(parser.output_path)
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, output_path=output_path, n_jobs=n_jobs)