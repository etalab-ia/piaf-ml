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

def concat_ID_and_name (id, name):
    return id + '//' + name


def iter_fiches(study_root):
    fiches = []
    for fiche in study_root.iter('Fiche'):
        fiche_id = fiche.attrib['ID']
        fiche_name = fiche.text
        if fiche_name == '\n': #sometimes there are fiches that do not belong to the theme
            continue
        fiche_txt = concat_ID_and_name(fiche_id, fiche_name)
        fiches.append(fiche_txt)
    return fiches


def extract_arbo_F_doc(doc_path):
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
        theme = root.find('Theme').find('Titre').text
        theme_id = root.find('Theme').attrib['ID']
        theme_txt = concat_ID_and_name(theme_id, theme)
    except :
        theme = ''

    # Cinq cas différents (cas de base + 4 types de fiches spéciaux), on les traite différemment
    try:
        dossier = root.find('DossierPere').find('Titre').text
        dossier_id = root.find('DossierPere').attrib['ID']
        dossier_txt = concat_ID_and_name(dossier_id, dossier)
    except: #why ??
        dossier_txt = list(root.findall(".//dc:type", namespaces={'dc': 'http://purl.org/dc/elements/1.1/'}))[0].text
        if dossier_txt in ['Question-réponse', 'Comment faire si...']:
            theme_txt = ''
        elif dossier_txt == 'Dossier':
            pass
        elif dossier_txt == 'Theme':
            return None
    # Sous-themes pas toujours présents
    try:
        sous_theme = root.find('SousThemePere').text
    except AttributeError:
        sous_theme = ''
    # Sous-dossiers pas toujours présents
    sous_dossier_dict = {}
    try:
        sous_dossier = list(root.iter("SousDossierPere"))[0].text
    except IndexError:
        sous_dossier = ''
    sous_dossier_dict[sous_dossier] = [f'{fiche}//{titre}']
    dossier_dict = {dossier_txt: sous_dossier_dict}
    sous_theme_dict = {sous_theme: dossier_dict}
    theme_dict = {theme_txt: sous_theme_dict}
    return theme_dict


def extract_arbo_N_doc(doc_path):
    """
    Extracts arbo from a single N doc in order to get the part theme / sous_theme / dosser
    """

    tree = ET.parse(doc_path)
    root = tree.getroot()
    #for sous_theme in list(root.iter("SousTheme")):

    try:
        type = root.attrib['type']
    except:
        print("not correct file -->" + doc_path)
        return {}

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

def fill_arborescence_with_arbo_file (arborescence, arbo):
    theme = list(arbo.keys())[0]
    if theme not in arborescence.keys():
        arborescence[theme] = arbo[theme]
    else:
        sous_theme = list(arbo[theme].keys())[0]
        if sous_theme not in arborescence[theme].keys():
            arborescence[theme][sous_theme] = arbo[theme][sous_theme]
        else:
            dossier = list(arbo[theme][sous_theme].keys())[0]
            if dossier not in arborescence[theme][sous_theme].keys():
                arborescence[theme][sous_theme][dossier] = arbo[theme][sous_theme][dossier]
            else:
                sous_dossier = list(arbo[theme][sous_theme][dossier].keys())[0]
                if sous_dossier not in arborescence[theme][sous_theme][dossier].keys():
                    arborescence[theme][sous_theme][dossier][sous_dossier] = arbo[theme][sous_theme][dossier][sous_dossier]
                else:
                    arborescence[theme][sous_theme][dossier][sous_dossier] += arbo[theme][sous_theme][dossier][sous_dossier]
    return arborescence


def fill_arborescence_with_N_files (doc_paths):
    arborescence = {}
    for doc_path in tqdm(doc_paths):
        arbo = extract_arbo_N_doc(doc_path)
        if len(arbo) != 0:
            arborescence = fill_arborescence_with_arbo_file(arborescence,arbo)
    return arborescence


def get_list_doc_in_arbo (arborescence):
    list_doc_in_arbo = []
    for theme in arborescence.keys():
        for sstheme in arborescence[theme].keys():
            for dossier in arborescence[theme][sstheme].keys():
                for ssdossier in arborescence[theme][sstheme][dossier].keys():
                    list_doc_in_arbo += arborescence[theme][sstheme][dossier][ssdossier]
    clean_list_doc_in_arbo = []
    for doc in list_doc_in_arbo:
        clean_list_doc_in_arbo.append(doc.split("//")[0])
    return clean_list_doc_in_arbo


def is_doc_in_arbo(doc_path, list_doc_arbo):
    doc_in_arbo = doc_path.stem in list_doc_arbo
    return doc_in_arbo


def complete_arbo_with_F_files (arborescence, doc_paths):
    list_doc_in_arbo = get_list_doc_in_arbo(arborescence)
    for doc_path in tqdm(doc_paths):
        if not is_doc_in_arbo(doc_path, list_doc_in_arbo):
            arbo_doc = extract_arbo_F_doc(doc_path)
            arborescence = fill_arborescence_with_arbo_file(arborescence, arbo_doc)
    return arborescence


def init_theme(theme):
    if theme == '':
        id = None
        name = 'Autre'
    else:
        id = theme.split('//')[0]
        name = theme.split('//')[1]
    return {'id': id,
            'type': 'theme',
            'name': name,
            'data': []
            }


def init_sous_theme(sous_theme):
    if sous_theme == "":
        name = 'Autre'
    else:
        name = sous_theme
    dict = {'id': None,
            'type': 'sous_theme',
            'name': name,
            'data': []
            }
    return dict

def init_dossier(dossier):
    if dossier == '':
        id = None
        name = 'Autre'
    elif '//' in dossier:
        id = dossier.split('//')[0]
        name = dossier.split('//')[1]
    else:
        id = None
        name = dossier
    return {'id': id,
            'type': 'dossier',
            'name': name,
            'data': []
            }


def init_sous_dossier(sous_dossier):
    if sous_dossier == "":
        name = 'Autre'
    else:
        name = sous_dossier
    dict = {'id': None,
            'type': 'sous_dossier',
            'name': name,
            'data': []
            }
    return dict


def init_fiche(fiche):
    if fiche == '':
        id = None
        name = 'Autre'
    else:
        id = fiche.split('//')[0]
        name = fiche.split('//')[1]
    return {'id': id,
            'type': 'fiche',
            'name': name,
            'data': None
            }


def reformat_json(arborescence):
    all_data = []
    for theme in arborescence.keys():
        theme_dict = init_theme(theme)
        sous_theme_exists = len(arborescence[theme].keys()) > 1
        theme_data = []
        for sous_theme in arborescence[theme].keys():
            if sous_theme_exists:
                sous_theme_dict = init_sous_theme(sous_theme)
            sous_theme_data = []
            for dossier in arborescence[theme][sous_theme]:
                dossier_dict = init_dossier(dossier)
                dossier_data = []
                sous_dossier_exists = len(arborescence[theme][sous_theme][dossier].keys()) > 1
                for sous_dossier in arborescence[theme][sous_theme][dossier].keys():
                    if sous_dossier_exists:
                        sous_dossier_dict = init_sous_dossier(sous_dossier)
                    sous_dossier_data = []
                    for fiche in arborescence[theme][sous_theme][dossier][sous_dossier]:
                        fiche_dict = init_fiche(fiche)
                        sous_dossier_data += [fiche_dict]
                    if sous_dossier_exists:
                        sous_dossier_dict['data'] = sous_dossier_data
                        dossier_data += [sous_dossier_dict]
                if not sous_dossier_exists:
                    dossier_data = sous_dossier_data
                dossier_dict['data'] = dossier_data
                sous_theme_data += [dossier_dict]
            if sous_theme_exists:
                sous_theme_dict['data'] = sous_theme_data
                theme_data += [sous_theme_dict]
        if not sous_theme_exists:
            theme_data = sous_theme_data
        theme_dict['data'] = theme_data
        all_data += [theme_dict]
    return {'version': '1.0',
            'data': all_data}



def main(doc_files_path, output_path, n_jobs):
    doc_files_path = Path(doc_files_path)
    output_path = Path(output_path)
    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        # go through the F files that contains the arbo
        doc_paths_F = glob(doc_files_path.as_posix() + "/**/F*.xml", recursive=True)
        doc_paths_F = [Path(p) for p in doc_paths_F]
        # go through the N files that contains the arbo
        doc_paths_N = glob(doc_files_path.as_posix() + "/**/N*.xml", recursive=True)
        doc_paths_N = [Path(p) for p in doc_paths_N]
    if not doc_paths_N:
        raise Exception(f"Path {doc_paths} not found")

    path = output_path / 'arborescence.json'
    arborescence = fill_arborescence_with_N_files(doc_paths_N)
    arborescence = complete_arbo_with_F_files(arborescence, doc_paths_F)

    arborescence = reformat_json(arborescence)

    if not output_path.exists():
        os.makedirs(output_path)



    with open(path.as_posix(), "w", encoding='utf-8') as out_file:
        json.dump(arborescence, out_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = parser.file_path
    output_path = parser.output_path
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, output_path=output_path, n_jobs=n_jobs)