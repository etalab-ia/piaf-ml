import json
import os

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path

notheme_sparse = json.load(open("./results/ea7a_detailed_results.json"))
notheme_dense = json.load(open("./results/4fd9_detailed_results.json"))

theme_sparse = json.load(open("./results/24bf_detailed_results.json"))
theme_dense = json.load(open("./results/3921_detailed_results.json"))

dossier_sparse = json.load(open("./results/d4ad_detailed_results.json"))
dossier_dense = json.load(open("./results/4278_detailed_results.json"))


def read_experiment(xp_name):
    df_results = pd.read_csv('./results/results.csv')
    retriever_type = df_results.loc[df_results['experiment_id'] == xp_name]['retriever_type'].values[0]
    filter_level = df_results.loc[df_results['experiment_id'] == xp_name]['filter_level'].values[0]
    if type(filter_level) != 'str': #in case there is no filter there will be a nan
        filter_level = 'None'
    lemma_preprocessing = df_results.loc[df_results['experiment_id'] == xp_name]['lemma_preprocessing'].values[0]
    meta = {
        'retriever_type': retriever_type,
        'filter_level': filter_level,
        'lemma_preprocessing': lemma_preprocessing,
    }
    dict = {
        'data': json.load(open(f"./results/{xp_name}_detailed_results.json")),
        'meta': meta
    }
    return dict


df = pd.DataFrame(columns=[
    'question',
    'fiche',
    'fiche_ok',
    'score_dense_no_lemma',
    'score_sparse_no_lemma',
    'score_dense_lemma',
    'score_sparse_lemma',
    'position_dense_no_lemma',
    'position_sparse_no_lemma',
    'position_dense_lemma',
    'position_sparse_lemma',
    'level'
])


def get_fiche_name(fiche_info):
    return fiche_info[0].split('--')[0]


def was_fiche_already_found(df, question, fiche_info, meta):
    try:
        was_fiche_found = df[(df.question == question) & (df.fiche == get_fiche_name(fiche_info)) & (
                    df.level == meta['filter_level'])].shape[
                              0] == 1
    except:
        print('was_fiche_already_found')
    return was_fiche_found



def add_info_to_df(df, question, fiche_info, meta):
    retriever_type = meta['retriever_type']
    if meta['lemma_preprocessing']:
        lemma_preprocessing = 'lemma'
    else:
        lemma_preprocessing = 'no_lemma'
    df.loc[(df.question == question) & (
            df.fiche == get_fiche_name(fiche_info)) & (
            df.level == meta['filter_level']), [f'score_{retriever_type}_{lemma_preprocessing}']] = fiche_info[2]
    df.loc[(df.question == question) & (
            df.fiche == get_fiche_name(fiche_info)) & (
            df.level == meta['filter_level']), f'position_{retriever_type}_{lemma_preprocessing}'] = fiche_info[1]
    return df


def add_fiche_to_df(df, question, fiche_info, fiche_ok, meta):
    try:
        data = [question, get_fiche_name(fiche_info), fiche_ok, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, meta['filter_level']]
        df.loc[len(df)] = data
    except:
        print('add_fiche_to_df')
    return df


def read_json_detailed_results(file, df):
    meta = file['meta']
    file = file['data']
    for result in ['successes', 'errors']:
        for question_id in file[result].keys():
            question_data = file[result][question_id]
            true_fiches = question_data['true_fiches']
            #question = question_data['question']
            for fiche_info in question_data['pred_fiches']:
                fiche_ok = get_fiche_name(fiche_info) in true_fiches
                if was_fiche_already_found(df, question_id, fiche_info, meta):
                    df = add_info_to_df(df, question_id, fiche_info, meta)
                else:
                    df = add_fiche_to_df(df, question_id, fiche_info, fiche_ok, meta)
                    df = add_info_to_df(df, question_id, fiche_info, meta)
    return df


def get_data_from_experience(dense_json, sparse_json):
    """
    This function takes two json and check the success and errors in common
    """
    data = []
    results_list = ['successes', 'errors']
    for results in results_list:
        for question in dense_json[results].keys():
            if question in list(sparse_json[results].keys()):
                try:
                    data.append([dense_json[results][question]['pred_fiches'][0][2],
                                 sparse_json[results][question]['pred_fiches'][0][2],
                                 results])
                except:
                    continue

    return pd.DataFrame(data, columns=['score_dense', 'score_sparse', 'result'])


def get_data_from_experience(dense_json, sparse_json):
    """
    This function takes two json and check the success and errors in common
    """
    data = []
    results_list = ['successes', 'errors']
    for results in results_list:
        for question in dense_json[results].keys():
            if question in list(sparse_json[results].keys()):
                try:
                    data.append([dense_json[results][question]['pred_fiches'][0][2],
                                 sparse_json[results][question]['pred_fiches'][0][2],
                                 results])
                except:
                    continue

    return pd.DataFrame(data, columns=['score_dense', 'score_sparse', 'result'])


def get_detailled_data_from_experience(dense_json, sparse_json):
    """
    This function takes two json and check the success and errors in common
    """
    data = []
    results = 'successes'
    for question in dense_json[results].keys():
        if question in list(sparse_json[results].keys()):
            # The successful answers that are in common
            try:
                data.append([dense_json[results][question]['pred_fiches'][0][2],
                             sparse_json[results][question]['pred_fiches'][0][2],
                             'success for both'])
            except:
                continue
        else:
            try:
                data.append([dense_json[results][question]['pred_fiches'][0][2],
                             sparse_json['errors'][question]['pred_fiches'][0][2],
                             'success for dense only'])
            except:
                continue
    for question in sparse_json[results].keys():
        if question not in list(dense_json[results].keys()):
            try:
                data.append([dense_json['errors'][question]['pred_fiches'][0][2],
                             sparse_json[results][question]['pred_fiches'][0][2],
                             'success for sparse only'])
            except:
                continue
    results = 'errors'
    for question in dense_json[results].keys():
        if question in list(sparse_json[results].keys()):
            try:
                data.append([dense_json[results][question]['pred_fiches'][0][2],
                             sparse_json[results][question]['pred_fiches'][0][2],
                             'errors for both'])
            except:
                continue

    return pd.DataFrame(data, columns=['score_dense', 'score_sparse', 'result'])


def save_results(result_file_path, df_results):
    if result_file_path.exists():
        df_old = pd.read_csv(result_file_path)
        df_results = pd.concat([df_old, df_results])
    else:
        if not result_file_path.parent.exists():
            os.makedirs(result_file_path.parent)
    with open(result_file_path.as_posix(), "w") as filo:
        df_results.to_csv(filo, index=False)


# df_res = get_detailled_data_from_experience(dossier_dense, dossier_sparse)
# df_res = pd.concat([df_res, get_detailled_data_from_experience(theme_dense, theme_sparse)])
# df_res = pd.concat([df_res, get_detailled_data_from_experience(dossier_dense, dossier_sparse)])

"""
for res in df_res.result.unique():
    x = df_res[df_res.result == res]['score_dense']
    y = df_res[df_res.result == res]['score_sparse']
    plt.scatter(x, y, label=res)
plt.legend()
plt.xlabel('score_dense')
plt.ylabel('score_sparse')
plt.show()"""

list_xp = [
    "690f",
    "0c8a",
    "6718",
    "b0e9",
    "c5e0",
    "4707",
    "7ead",
    "d1b9",
    "073d",
    "34f4",
    "5007",
    "d281"
]

for xp_name in tqdm(list_xp):
    file = read_experiment(xp_name)
    df = read_json_detailed_results(file, df)

save_results(Path('./results/analysis_results.csv'), df)

print('hello')
