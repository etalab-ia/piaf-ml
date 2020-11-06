import pandas as pd
import json
from pathlib import Path

"""
{
  "version": "v2.0",
  "data": [{
    "title": "Sport",
    "paragraphs": [{
      "context": "La Gr\u00e8ce, Rome, Byzance, l'Occident m\u00e9di\u00e9val puis moderne, mais aussi l'Am\u00e9rique pr\u00e9colombienne  ou l'Asie, sont tous marqu\u00e9s par l'importance du sport. Certaines p\u00e9riodes sont surtout marqu\u00e9es par des interdits concernant le sport, comme c'est le cas en Grande-Bretagne du Moyen \u00c2ge \u00e0 l'\u00e9poque Moderne. Interrog\u00e9e sur la question, la Justice anglaise tranche ainsi en 1748 que le cricket n\u2019est pas un jeu ill\u00e9gal. Ce sport, comme tous les autres, figurait en effet sur des \u00e9dits royaux d'interdiction r\u00e9guli\u00e8rement publi\u00e9s par les monarques britanniques du XIe  au  XVe si\u00e8cle. En 1477, la pratique d'un \u00ab jeu interdit \u00bb est ainsi passible de trois ans de prison. Malgr\u00e9 l'interdit, la pratique perdure, n\u00e9cessitant un rappel quasi permanent \u00e0 la r\u00e8gle.",
      "qas": []
    }],
    "categorie": "Histoire",
    "displaytitle": "Premier combat naval de Tripoli",
    "wikipedia_page_id": 7138870
  }]
}"""


path_arbo = Path('./data/arborescence_spf_particuliers.json')
with open(path_arbo) as file:
    arborescence = json.load(file)

def get_arbo(arbo):
    res = {'theme': '', 'sous_theme': '', 'dossier': '', 'sous_dossier': '', 'fiche': '', 'id': ''}
    for level_dict in arbo:
        res[level_dict['type']] = level_dict['name']
        if level_dict['type'] == 'fiche':
            res['id'] = level_dict['id']
    return res

def get_arborescence(arborescence, fiche_id):
    arborescence = arborescence['data']
    for level_1_dict in arborescence:
        arbo = [level_1_dict]
        for level_2_dict in level_1_dict['data']:
            if len(arbo) > 1:
                arbo= arbo[:1]
            arbo.append(level_2_dict)
            for level_3_dict in level_2_dict['data']:
                if len(arbo) > 2:
                    arbo= arbo[:2]
                arbo.append(level_3_dict)
                if level_3_dict['id'] == fiche_id:
                    return get_arbo(arbo)
                elif level_3_dict['type'] == 'fiche':
                    continue
                else:
                    for level_4_dict in level_3_dict['data']:
                        if len(arbo) > 3:
                            arbo= arbo[:3]
                        arbo.append(level_4_dict)
                        if level_4_dict['id'] == fiche_id:
                            return get_arbo(arbo)
                        elif level_4_dict['type'] == 'fiche':
                            continue
                        else:
                            try:
                                for level_5_dict in level_4_dict['data']:
                                    if len(arbo) > 4:
                                        arbo= arbo[:4]
                                    arbo.append(level_5_dict)
                                    if level_5_dict['id'] == fiche_id:
                                        return get_arbo(arbo)
                            except:
                                print('hello')

def add_data_to_list(data_question, data_list, context, qas):
    if len(data_list) == 0:
        data_question['paragraphs'] = [{'context': context, 'qas': [qas]}]
        data_list = [data_question]
    else:
        for fiche in data_list:
            if fiche['id'] == data_question['id']:
                fiche['paragraphs'][0]['qas'].append(qas)
            else:
                data_question['paragraphs'] = [{'context': context, 'qas': [qas]}]
        data_list.append(data_question)
    return data_list

def get_context_from_url(fiche_id):
    path_fiche = Path('./data/v10') / f'{fiche_id}.json'
    with open(path_fiche) as file:
        file_content = json.load(file)
    return file_content['text']

df = pd.read_csv('./data/407_question-fiche_anonym.csv')
dj_json = df.to_json(orient='records')
dataset = json.loads(dj_json)
data_list = []
for question in dataset:
    qas = question['incoming_message']
    url = question['url']
    fiche_id = url.split('/')[-1]
    try:
        context = get_context_from_url(fiche_id)
        data_question = get_arborescence(arborescence, fiche_id)
        data_list = add_data_to_list(data_question, data_list, context, qas)
    except:
        print(f'the fiche {fiche_id} does not exists !!')
        context = 'none'

print('hello')

