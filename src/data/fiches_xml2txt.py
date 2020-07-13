'''
Transforms service public france fiches in XML format to txt. It tries to extract the essential content from the fiches

Usage:
    fiches_xml2txt.py <file_path> <output_path> [options]

Arguments:
    <file_path>             A path of a single XML fiche or a folder with multiple fiches XML
    <output_path>           A path where to store the extracted info
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm

from src.util.files import slugify

TYPE_FICHES = ["associations", "particuliers", "entreprise"]
ERROR_COUNT = 0

def save_subfiche(doc_path: Path, situation_title: str, chapitre_title: str, cas_title: str,
                  output_path: Path, cas_text: List[str]):
    fiche_type = [t for t in TYPE_FICHES if t in doc_path.as_posix()][0]
    output_path = output_path / fiche_type

    file_name = doc_path.stem
    new_dir_subfiche_path = output_path / file_name
    if not new_dir_subfiche_path.exists():
        new_dir_subfiche_path.mkdir()
    subfiche_file_name = f"{file_name}_{slugify(situation_title)[:15]}--{slugify(chapitre_title)[:15]}_--" \
                         f"{slugify(cas_title)[:15]}.txt"

    new_fiche_path = new_dir_subfiche_path / subfiche_file_name
    tqdm.write(f"\tSaving sub-fiche to {new_fiche_path}")
    with open(new_fiche_path.as_posix(), "w") as subfiche:
        subfiche.write(situation_title + "\n\n")
        subfiche.write(chapitre_title + "\n\n")
        # subfiche.write(cas_title + "\n")
        subfiche.write("\n".join(cas_text))

    pass

def try_get_text(root, tag):
    existing_tags = list(root.iter(tag))
    if not existing_tags:
        return ""
    else:
        text = list(existing_tags[0].itertext())
        return text

def run(doc_path: Path, output_path: Path):
    global ERROR_COUNT
    try:
        tqdm.write(f"Extracting info from {doc_path}")
        tree = ET.parse(doc_path)
        root = tree.getroot()

        introduction_text = try_get_text(root, "Introduction")
        situations = list(root.iter("Situation"))


        if not situations:
            tqdm.write(f"\tCould not treat file {doc_path}. It has no situations !")
            return 0
        for situation in situations:
            situation_text = try_get_text(root, "Situation")
            situation_title = situation.find("Titre").text  # kinda hacky :/
            for chapitre in situation.iter("Chapitre"):
                chapitre_title = list(chapitre.iter("Titre"))[0].find("Paragraphe").text
                cass = list(chapitre.iter("Cas"))
                if not cass:
                    continue  #  This is wrong. Even if we dont have cases we have text
                for cas in cass:
                    # cas_title = cas.find("Titre").find("Paragraphe").text
                    cas_title = list(cas.iter("Titre"))[0].find("Paragraphe").text
                    cas_text = []
                    for cas_paragraph in cas.iter("Paragraphe"):
                        cas_text.append(list(cas_paragraph.itertext()))
                    cas_text = [" ".join(t) for t in cas_text]
                    save_subfiche(doc_path=doc_path, situation_title=situation_title, chapitre_title=chapitre_title,
                                  cas_title=cas_title, output_path=output_path, cas_text=cas_text)

        return 1
    except Exception as e:
        tqdm.write(f"Could not treat file {doc_path}. Error: {str(e)}")
        # raise
        ERROR_COUNT += 1
        return 0


def main(doc_files_path: Path, output_path: Path, n_jobs: int):
    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path.as_posix() + "/**/F*.xml", recursive=True)
        doc_paths = [Path(p) for p in doc_paths]
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if n_jobs < 2:
        job_output = []
        for doc_path in tqdm(doc_paths):
            tqdm.write(f"Converting file {doc_path}")
            job_output.append(run(doc_path, output_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(run)(doc_path, output_path) for doc_path in tqdm(doc_paths))
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
    main(doc_files_path=doc_files_path, output_path=output_path, n_jobs=n_jobs)
