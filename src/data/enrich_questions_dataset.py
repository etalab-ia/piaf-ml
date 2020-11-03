from pathlib import Path
import pandas as pd
import json

path_dataset = Path('data/407_question-fiche_anonym.csv')
path_arborescence = Path('data/arborescence.json')

df = pd.read_csv(path_dataset)
with open(path_arborescence) as f:
  arborescence = json.load(f)

def get_fiche_info(fiche):
    res = {}
    for level_1 in arborescence['data']:
        type_l1 = level_1['type']
        name_l1 = level_1['name']
        res[type_l1] = name_l1
        for level_2 in level_1['data']:
            type_l1 = level_1['type']
            name_l1 = level_1['name']
            res[type_l1] = name_l1