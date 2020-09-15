'''
Transforms service public france fiches in XML format to txt. It tries to extract the essential content from the fiches

Usage:
    fiches_xml2txt.py <file_path> <output_path> [options]

Arguments:
    <file_path>             A path of a single XML fiche or a folder with multiple fiches XML
    <output_path>           A path where to store the extracted info
    --cores=<n> CORES       Number of cores to use [default: 1:int]
    --as_json=<j> AS_JSON      Whether or not output JSON files instead of TXT [default: 0:int]
    --as_one=<j> AS_ONE     Whether or not output 1 file for 1 TXT, or consider sub-files
'''
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

TYPE_FICHES = ["associations", "particuliers", "entreprise"]
ERROR_COUNT = 0


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def save_subfiche(doc_path: Path,
                  output_path: Path,
                  fiche_title: str = None, fiche_intro_text: str = None,
                  situation_title: str = None, situation_text: str = None,
                  chapitre_title: str = None, chapitre_text: str = None,
                  cas_title: str = None, cas_text: str = None,
                  create_folders: bool =False, as_json: bool = False):
    # fiche_type = [t for t in TYPE_FICHES if t in doc_path.as_posix()][0]
    # output_path = output_path / fiche_type

    file_name = doc_path.stem
    subfiche_file_name = f"{file_name}"
    if situation_title:
        subfiche_file_name += f"--{slugify(situation_title)[:10]}-{slugify(situation_title)[-6:]}-{len(situation_title)}"
    if chapitre_title:
        subfiche_file_name += f"--{slugify(chapitre_title)[:10]}-{slugify(chapitre_title)[-6:]}-{len(chapitre_title)}"
    if cas_title:
        subfiche_file_name += f"--{slugify(cas_title)[:10]}-{slugify(cas_title)[-6:]}-{len(cas_title)}"

    subfiche_file_name += ".txt" if not as_json else ".json"

    if create_folders:
        new_dir_subfiche_path = output_path / file_name
        if not new_dir_subfiche_path.exists():
            new_dir_subfiche_path.mkdir()
        new_fiche_path = new_dir_subfiche_path / subfiche_file_name
    else:
        new_fiche_path = output_path / subfiche_file_name

    tqdm.write(f"\tSaving sub-fiche to {new_fiche_path}")
    subfiche_string = f"{fiche_title} : "
    if fiche_intro_text:
        subfiche_string += f"{fiche_intro_text}"

    subfiche_string += "\n\n"

    if situation_title:
        subfiche_string += f"{situation_title}: "
    if situation_text:
        if not (slugify(cas_text.lower().strip()) in slugify(situation_text.lower().strip())):
            print('Entered')
            subfiche_string += f"{situation_text}"
        subfiche_string += "\n\n"

    if chapitre_title:
        subfiche_string += f"{chapitre_title}: "
    if chapitre_text:
        subfiche_string += f"{chapitre_text}"
        subfiche_string += "\n\n"

    if cas_text:
        if not (slugify(cas_text.lower().strip()) in slugify(subfiche_string.lower().strip())):
            subfiche_string += f"{cas_text.lstrip(chapitre_title)}"

    if not as_json:
        with open(new_fiche_path.as_posix(), "w", encoding='utf-8') as subfiche:
            subfiche.write(subfiche_string)
    else:
        with open(new_fiche_path.as_posix(), "w", encoding='utf-8') as subfiche:
            content = {'text': subfiche_string,
                       'link': f'https://www.service-public.fr/particuliers/vosdroits/{file_name}'}
            json.dump(content, subfiche, indent=4, ensure_ascii=False)


def save_fiche_as_one(doc_path: Path,
                  output_path: Path,
                  fiche_title: str = None, fiche_text: str = None,
                  create_folders: bool =False, as_json: bool = False):

    new_fiche_path = doc_path.stem

    extension = ".txt" if not as_json else ".json"
    new_fiche_path += extension
    new_fiche_path = output_path / new_fiche_path

    tqdm.write(f"\tSaving entire fiche to {new_fiche_path}")
    fiche_string = f"{fiche_title} : "
    fiche_string += "\n\n"
    fiche_string += fiche_text

    if not as_json:
        with open(new_fiche_path.as_posix(), "w", encoding='utf-8') as newfiche:
            newfiche.write(fiche_string)
    else:
        with open(new_fiche_path.as_posix(), "w", encoding='utf-8') as newfiche:
            content = {'text': fiche_string,
                       'link': f'https://www.service-public.fr/particuliers/vosdroits/{file_name}'}
            json.dump(content, newfiche, indent=4, ensure_ascii=False)



def try_get_text(root: Element, tag: str) -> str:
    existing_tags = list(root.iter(tag))
    if not existing_tags:
        return ""
    else:
        text = list(existing_tags[0].itertext())
        text = " ".join(text)
        return text


def try_get_situation_text(child: Element,
                           list_tags_to_remove: List[str] = ["Titre", "BlocCas", "SousChapitre"]) -> str:
    """
    The chapitres text are apparently in the first children level of the Chapitre as a Paragraphe(s).
    So the idea here is to grab all the paragraphs that occur before anything else (a BlocCas p. ex) and consider it as
    the chapitre text (all the text that comes before the Cases)
    Given that all text (apparently) appears in Paragraphe tags even if they are within Lists, so we remove the
    Titre and BlocCas to avoid these texts
    :param list_tags_to_remove: we want to keep all the paragraphs except those inside Titre and BlocCas, so we pop them

    :param child:
    :return:
    """
    situation_children = list([t for t in child])
    tags = [t.tag for t in situation_children]
    filtered_chapitre_children = [situation_children[i] for i, t in enumerate(tags) if t not in list_tags_to_remove]

    chapitre_text_list = []
    for child in filtered_chapitre_children:
        for cas_paragraph in child.iter("Paragraphe"):
            chapitre_text_list.append(list(cas_paragraph.itertext()))

    chapitre_text_list = [" ".join(t) for t in chapitre_text_list]
    if chapitre_text_list:
        chapitre_text = " ".join(chapitre_text_list).replace("\n", " ")
        return chapitre_text
    else:
        return ""



def try_get_situation_title(child: Element) -> str:
    situation_title = ""
    titre = [t for t in list(child) if t.tag == "Titre"]
    if titre:
        titre = titre[0]
        if hasattr(titre, "text"):
            situation_title = titre.text
            return situation_title
    return situation_title


def try_get_chapitre_title(child: Element) -> str:
    chapitre_title = ""
    titre = [t for t in list(child) if t.tag == "Titre"]
    if titre:
        titre = titre[0]
        paragraph_titre = [t for t in list(titre) if t.tag == "Paragraphe"]
        if paragraph_titre:
            chapitre_title = paragraph_titre[0].text
            return chapitre_title
    return chapitre_title


def try_get_chapitre_text(child: Element, list_tags_to_remove: List[str] = ["Titre", "BlocCas", "SousChapitre"]) -> str:
    """
    The chapitres text are apparently in the first children level of the Chapitre as a Paragraphe(s).
    So the idea here is to grab all the paragraphs that occur before anything else (a BlocCas p. ex) and consider it as
    the chapitre text (all the text that comes before the Cases)
    Given that all text (apparently) appears in Paragraphe tags even if they are within Lists, so we remove the
    Titre and BlocCas to avoid these texts
    :param list_tags_to_remove:
    :param child:
    :return:
    """
    chapitre_children = list([t for t in child])
    tags = [t.tag for t in chapitre_children]
    # we want to keep all the paragraphs except those inside Titre and BlocCas, so we pop them
    filtered_chapitre_children = [chapitre_children[i] for i, t in enumerate(tags) if t not in list_tags_to_remove]

    chapitre_text_list = []
    for child in filtered_chapitre_children:
        for cas_paragraph in child.iter("Paragraphe"):
            chapitre_text_list.append(list(cas_paragraph.itertext()))

    chapitre_text_list = [" ".join(t) for t in chapitre_text_list]
    if chapitre_text_list:
        chapitre_text = " ".join(chapitre_text_list).replace("\n", " ")
        return chapitre_text
    else:
        return ""


def try_get_title_cas(cas):
    try:
        cas_title = list(cas.iter("Titre"))[0].find("Paragraphe").text
        return cas_title
    except:
        return ""


def try_get_title_fiche(child):
    try:
        fiche_tile = list(list(child.iter("Publication"))[0])[0].text
        return fiche_tile
    except:
        return ""


def treat_no_situation_fiche(root: Element):
    # it is a fiche without situations

    fiche_text = ""
    fiche_text_list = []
    for paragraph in root.iter("Paragraphe"):
        fiche_text_list.append(list(paragraph.itertext()))
    fiche_text_list = [" ".join(t) for t in fiche_text_list]
    if fiche_text_list:
        fiche_text = "\n".join(fiche_text_list)

    if fiche_text:
        return "\n" + fiche_text
    else:
        # raise Exception("Fiche without situation. We could not extract anything")
        return


def clean_elements(root: Element):
    """
    Keep the interesting bits of the XML (and hence the essential text of the fiche)
    :param root:
    :return:
    """
    tags = [t for t in list(root)]
    tag_names = [t.tag for t in tags]
    interesting_tags = ["ListeSituations"]
    # 1. First remove the http namespaced elements
    tags = [tags[i] for i, t in enumerate(tag_names) if "http" not in t]


def run(doc_path: Path, output_path: Path, as_json: bool):
    global ERROR_COUNT
    has_situations = True
    has_chapitres = True
    has_cases = True
    try:
        tqdm.write(f"Extracting info from {doc_path}")
        tree = ET.parse(doc_path)
        root = tree.getroot()
        fiche_title = list(list(root.iter("Publication"))[0])[0].text
        introduction_text = try_get_text(root, "Introduction")
        situations = list(root.iter("Situation"))

        if not situations:
            #     tqdm.write(f"\tFile {doc_path} has no situations !")
            #     subfiche_text = treat_no_situation_fiche(root)
            #     if subfiche_text:
            #         save_subfiche(doc_path=doc_path, output_path=output_path, fiche_title=fiche_title,
            #                       cas_text=subfiche_text)
            #
            #         return 1
            #     else:
            #         return 0
            alternative_tags = ["Texte", "Introduction"]
            for alt in alternative_tags:
                found_tag = root.find(alt)
                if root.find(alt):
                    situations = [found_tag]
                    break
            has_situations = False
        for situation in situations:
            situation_text = try_get_situation_text(situation)
            situation_title = try_get_situation_title(situation)  # kinda hacky :/
            chapitres = list(situation.iter("Chapitre"))
            if not chapitres:
                chapitres = [situation]
                has_chapitres = False
            for chapitre in chapitres:
                # chapitre_title = list(chapitre.iter("Titre"))[0].find("Paragraphe").text
                chapitre_text, chapitre_title = "", ""
                if has_chapitres:
                    chapitre_title = try_get_chapitre_title(chapitre)
                    chapitre_text = try_get_chapitre_text(chapitre)
                cases = list(chapitre.iter("Cas"))
                if not cases:
                    #  There aren't any cases, so we treat the chapitre element as the single case. This is hacky
                    cases = [chapitre]
                    has_cases = False
                for cas in cases:
                    # cas_title = cas.find("Titre").find("Paragraphe").text
                    if len(cases) == 1:
                        cas_title = ""
                    else:
                        cas_title = try_get_title_cas(cas)
                    cas_text_list = []
                    for cas_paragraph in cas.iter("Paragraphe"):
                        cas_text_list.append(list(cas_paragraph.itertext()))
                    cas_text_list = [" ".join(t) for t in cas_text_list]
                    cas_text = "\n".join(cas_text_list)
                    save_subfiche(doc_path=doc_path,
                                  fiche_title=fiche_title, fiche_intro_text=introduction_text,
                                  situation_title=situation_title, situation_text=situation_text,
                                  chapitre_title=chapitre_title, chapitre_text=chapitre_text,
                                  cas_title=cas_title, cas_text=cas_text,
                                  output_path=output_path,
                                  as_json=as_json,
                                  )

        return 1
    except Exception as e:
        tqdm.write(f"Could not treat file {doc_path}. Error: {str(e)}")
        raise
        ERROR_COUNT += 1
        return 0

def run_fiche_as_one(doc_path: Path, output_path: Path, as_json: bool):
    global ERROR_COUNT
    fiche_text = ""

    try:
        tqdm.write(f"Extracting info from {doc_path}")
        tree = ET.parse(doc_path)
        root = tree.getroot()
        fiche_title = list(list(root.iter("Publication"))[0])[0].text
        fiche_text += treat_no_situation_fiche(root)

        save_fiche_as_one(doc_path=doc_path,
                      fiche_title=fiche_title,
                      fiche_text=fiche_text,
                      output_path=output_path,
                      as_json=as_json,
                      )
        return 1
    except Exception as e:
        tqdm.write(f"Could not treat file {doc_path}. Error: {str(e)}")
        raise
        ERROR_COUNT += 1
        return 0


def main(doc_files_path: Path, output_path: Path, as_json: bool, n_jobs: int, as_one: bool):
    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path.as_posix() + "/**/F*.xml", recursive=True)
        doc_paths += glob(doc_files_path.as_posix() + "/**/N*.xml", recursive=True)
        doc_paths = [Path(p) for p in doc_paths]
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if n_jobs < 2:
        job_output = []
        for doc_path in tqdm(doc_paths):
            tqdm.write(f"Converting file {doc_path}")
            if as_one:
                job_output.append(run_fiche_as_one(doc_path, output_path, as_json))
            else:
                job_output.append(run(doc_path, output_path, as_json))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(run)(doc_path, output_path, as_json)
                                             for doc_path in tqdm(doc_paths))
    tqdm.write(
        f"{sum(job_output)} XML fiche files were extracted to TXT. {len(job_output) - sum(job_output)} files "
        f"had some error.")
    tqdm.write(f"Error count {ERROR_COUNT}")
    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.file_path)
    output_path = Path(parser.output_path)
    n_jobs = parser.cores
    as_json = True if int(parser.as_json) > 0 else False
    as_one = True if int(parser.as_one) > 0 else False
    main(doc_files_path=doc_files_path, output_path=output_path, as_json=as_json, n_jobs=n_jobs, as_one=as_one)
