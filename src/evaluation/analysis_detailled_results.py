import json
import os

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path


def read_experiment(xp_name):
    df_results = pd.read_csv('./results/results.csv')
    retriever_type = df_results.loc[df_results['experiment_id'] == xp_name]['retriever_type'].values[0]
    filter_level = df_results.loc[df_results['experiment_id'] == xp_name]['filter_level'].values[0]
    if type(filter_level) != str: #in case there is no filter there will be a nan
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


def save_results(result_file_path, df_results):
    if result_file_path.exists():
        df_old = pd.read_csv(result_file_path)
        df_results = pd.concat([df_old, df_results])
    else:
        if not result_file_path.parent.exists():
            os.makedirs(result_file_path.parent)
    with open(result_file_path.as_posix(), "w") as filo:
        df_results.to_csv(filo, index=False)


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


list_xp = [
    'b850',
    '1ea4',
    'a0ef',
    '3a76',
    '862d',
    'ba96',
    '208c',
    '9e00'
]

for xp_name in tqdm(list_xp):
    file = read_experiment(xp_name)
    df = read_json_detailed_results(file, df)

save_results(Path('./results/analysis_results.csv'), df)

print('hello')
