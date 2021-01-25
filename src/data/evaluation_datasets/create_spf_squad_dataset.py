"""
WARNING: This script is a one-shot script.
This script creates a single SQuAD format SPF evaluation dataset. This file is the merge of these three elements:
1. The uni-fiche JSON files created from the original SPF xml files (these files contain the dossier,theme metadata)
2. The 105 annotated Question Fiches dataset (data/squad-style-datasets/spf_qr_test.json)
3. The ~530 Question Fiches dataset  (data/questions_spf.json)

The output should resemble this :
{
  "version": "v1.0",
  "data": [{
    'title': 'Immatriculer un véhicule d'occasione',
    'sous_theme': '',
    'categorie': "Carte grise (certificat d'immatriculation)",
    'sous_dossier': 'Immatriculer un véhicule',
    'reference': "Immatriculer un véhicule d'occasion",
    'id': 'F1050',
    'paragraphs': [{
        'context': "Carte grise\xa0:..."
        'qas': [
            {'question':"J'ai acheter ...?"
             'answers': [ ... ]
             "is_impossible": False

            }]
        }]
    }]

Usage:
    create_spf_squad_dataset.py <spf_fiches_folder> <annotated_questions_spf>

Arguments:
    <spf_fiches_folder>             A path where to find the list of [fiche_id, question, answer] to transform
    <annotated_questions_spf>       Path to the SQuAD JSON file with the annotated spf questions
"""

import glob
import json
from pathlib import Path

from argopt import argopt


def create_squad_dataset(spf_fiches_folder: Path,
                         annotated_questions_spf_path: Path):
    # 1. Read all the fiches JSON files
    spf_jsons_paths = [Path(f) for f in glob.glob(spf_fiches_folder.as_posix() + "/*.json")]
    dict_spf_jsons = {}
    for path in spf_jsons_paths:
        with open(path) as filo:
            dict_spf_jsons[path.stem] = json.load(filo)

    # 2. Read all the 105 answered questions + 525 non-answered questions from the annotated dataset
    # Get titles of aready answered questions and of all questions
    with open(annotated_questions_spf_path) as filo:
        annotated_questions_spf = json.load(filo)["data"]
    non_answered_question_fiches_ids = []
    answered_questions_fiches_ids = []
    dict_question_spf = {}
    for fiche in annotated_questions_spf:
        dict_question_spf[fiche["title"]] = fiche
        for paragraph in fiche["paragraphs"]:
            if "qas" in paragraph:
                answered_questions_fiches_ids.append(fiche["title"])
                continue
            else:
                non_answered_question_fiches_ids.append(fiche['title'])
    all_questions_fiches = answered_questions_fiches_ids + non_answered_question_fiches_ids

    non_question_fiches = [f for f in list(dict_spf_jsons.keys()) if f not in all_questions_fiches]
    # 3. Begin with the creation of a SQuAD forma dataset from SPF jsons
    spf_data = []
    for fiche_id in non_question_fiches + all_questions_fiches:
        fiche_content = dict_spf_jsons[fiche_id]
        # if fiche_id in all_questions_fiches:
        #     print("stop!")  # we will add question fiches at the end
        squad_dict = {"link": fiche_content["link"]}

        if len(fiche_content["text"]) < 30:
            print(f"Fiche {fiche_id} has text too small: {fiche_content['text']}")
            continue
        if "arborescence" in fiche_content and fiche_content["arborescence"]:
            squad_dict["title"] = fiche_content["arborescence"].pop("fiche")
            squad_dict.update(fiche_content["arborescence"])
        else:
            print(f"Fiche {fiche_id} has no arborescence")
            squad_dict["title"] = fiche_content["text"].split("\n")[0][:-3]
            squad_dict.update({
                'sous_theme': '',
                'categorie': '',
                'sous_dossier': '',
                'reference': squad_dict["title"],
                'id': fiche_id
            })

        squad_dict["paragraphs"] = [
            {
                "context": fiche_content["text"],
                "qas": [
                    {
                        "question": "" if fiche_id not in non_answered_question_fiches_ids else squad_dict["title"],
                        "answers": [
                            {"answer_start": -1, "text": ""}] if fiche_id not in answered_questions_fiches_ids else
                        dict_question_spf[fiche_id]["paragraphs"][0]["qas"][0]["answers"],
                        "is_impossible": False
                    }
                ]
            }
        ]
        spf_data.append(squad_dict)

    # 4. Save the new dataset
    new_dataset = {"version": 1.0,
                   "data": spf_data}
    with open(annotated_questions_spf_path.parent / Path("full_spf_squad.json"), "w") as filo:
        json.dump(new_dataset, filo, indent=4, ensure_ascii=False)


def main():
    parser = argopt(__doc__).parse_args()
    spf_fiches_folder = Path(parser.spf_fiches_folder)
    annotated_questions_spf_path = Path(parser.annotated_questions_spf)
    create_squad_dataset(spf_fiches_folder=spf_fiches_folder,
                         annotated_questions_spf_path=annotated_questions_spf_path)


if __name__ == '__main__':
    main()
