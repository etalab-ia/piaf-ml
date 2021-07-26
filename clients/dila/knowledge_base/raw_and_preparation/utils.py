import csv
import json

def add_question_to_squad():
    """
    We have a list of question that DILA prodvided for better autocomplete.
    To add them to the dataset, we created this simple script

    Once you are satisfied with your squad.json, simply run 
    add_question_to_squad()
    and you will add to your squad.json the question manully wirtten in the 
    manual_question_autocomplete.csv file
    """
    l = []
    with open('manual_question_autocomplete.csv', newline='') as csvfile:
        links_file = csv.DictReader(csvfile)
        for row in links_file:
            l.append(row)

    with open('squad_only_fiches.json') as f:
        squad = json.load(f)
        # for each fiche in the squad.json we will look the links   
        for fiche in squad["data"]:
            fiche_link = fiche["link"]
            # let's see if any question matches the links bellow
            for question_lien in l:
                if question_lien["Lien"] == fiche_link:
                    print('bingo')
                    qa = {
                    "question": question_lien["Question"],
                    "answers": [],
                    "is_impossible": False
                    }
                    fiche["paragraphs"][0]["qas"].append(qa)

    with open('../squad.json', 'w', encoding='utf-8') as f:
        json.dump(squad, f, ensure_ascii=False, indent=4)



def merge_json_files(squad_file_path,aide_file_path,demarches_file_path):
    """
    we will merge the three dataset for service-public:
    - Fiches coming from data.gouv.fr
    - Aide manually extracted
    - demarches manually extracted

    Simply run 
    merge_json_files('squad_only_fiches.json','manual_dataset_aide.json','manual_dataset_demarches.json')
    and you will output a new squad.json in the top parent folder (we overwrite the old squad.json)    
    """
    with open(squad_file_path) as f1:
        s = json.load(f1)
    with open(aide_file_path) as f2:
        a = json.load(f2)
    with open(demarches_file_path) as f3:
        d = json.load(f3)
    data = s["data"] + a["data"] + d["data"]
    s["data"] = data
    with open('../squad.json', 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=4)
