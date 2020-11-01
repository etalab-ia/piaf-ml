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
    {'dense': '1ea4',
     'sparse': 'b850'}
]


def read_questions(file):
    list_question = []
    for result in ['successes', 'errors']:
        for question_id in file[result].keys():
            list_question.append(question_id)
    return list_question

def get_true_fiches(file, question_id):
    for result in ['successes', 'errors']:
        if question_id in file[result].keys():
            true_fiches = file[result][question_id]['true_fiches']
    return true_fiches

def get_results(question_id, file):
    pred_fiches = []
    for result in ['successes', 'errors']:
        if question_id in file[result].keys():
            pred_fiches = (file[result][question_id]['pred_fiches'])
    return pred_fiches

def get_pred_fiches(retrieved_results, l):
    pred_fiches = []
    for res in retrieved_results[:l]:
        pred_fiches.append(res[0])
    return pred_fiches

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def get_list_common_fiches(retrieved_results_dense, retrieved_results_sparse, l):
    list_fiches_dense = get_pred_fiches(retrieved_results_dense, l)
    list_fiches_sparse = get_pred_fiches(retrieved_results_sparse, l)
    return intersection(list_fiches_dense, list_fiches_sparse)

def get_scores(list_fiches, retrieved_results, param_dense):
    scores = np.zeros(len(list_fiches))
    i = 0
    for res in retrieved_results:
        if res[0] in list_fiches:
            scores[i]=res[2]
    # CHANGE scores = (scores - param_dense['mean']) / param_dense['scale']
    if len(list_fiches) == 1:
        scores = np.array([scores])
    return  scores

def combine_dense_sparse(retrieved_results_dense, retrieved_results_sparse, k, l):
    param_sparse = {'mean':18.33259795, "scale":8.13641822}
    param_dense = {'mean': 0.41424847, "scale":0.10654837}
    list_common_fiches = get_list_common_fiches(retrieved_results_dense, retrieved_results_sparse, l)
    scores_dense = get_scores(list_common_fiches, retrieved_results_dense, param_dense)
    scores_sparse = get_scores(list_common_fiches, retrieved_results_sparse, param_sparse)
    score_final = np.sqrt(scores_dense * scores_sparse) # CHANGE np.abs(scores_dense * scores_sparse)
    res = pd.DataFrame([list_common_fiches, score_final.tolist()])
    res = res.transpose()
    res.columns = ['list_common_fiches', 'score_final']
    return res.sort_values('score_final', ascending=True).head(k)

def compute_retriever_precision(true_fiches, retrieved_results, weight_position=True):
    """
    Computes an accuracy-like score to determine the fairness of the retriever.
    Takes the k *retrieved* fiches' names and counts how many of them exist in the *true* fiches names


    :param retrieved_fiches:
    :param true_fiches:
    :param weight_position: Bool indicates if the precision must be calculated with a weighted precision
    :return:
    """
    retrieved_docs = []
    summed_precision = 0
    results_info = {}
    for fiche_idx, true_fiche_id in enumerate(true_fiches):
        if 'demande' in true_fiche_id:
            print('helel')
        print(f'True fiche os:{true_fiche_id}')
        for retrieved_doc_idx in range(retrieved_results.shape[0]):
            retrieved_doc = retrieved_results.iloc[retrieved_doc_idx,0]
            retrieved_docs.append(retrieved_doc)
            if true_fiche_id in retrieved_doc:
                if weight_position:
                    summed_precision += ((fiche_idx + 1) / (fiche_idx + retrieved_doc_idx + 1))
                else:
                    summed_precision += 1
                break
        if len(retrieved_docs) > 10:
            print(len(retrieved_docs))


    return summed_precision, results_info

for xp_name in tqdm(list_xp):
    summed_precision = 0
    nb_questions = 0
    json_dense = read_experiment(xp_name['dense'])['data']
    json_sparse = read_experiment(xp_name['sparse'])['data']
    questions = read_questions(json_dense)
    for question in questions:
        true_fiche_urls = get_true_fiches(json_dense, question)
        true_fiche_ids = [f.split("/")[-1] for f in true_fiche_urls]
        retrieved_results_dense = get_results(question, json_dense)
        retrieved_results_sparse = get_results(question, json_sparse)

        retrieved_results = combine_dense_sparse(retrieved_results_sparse, retrieved_results_sparse,20, 100)
        if retrieved_results.shape[0] > 10:
            print('hell')
        precision, results_info = compute_retriever_precision(true_fiche_ids,
                                                              retrieved_results,
                                                              weight_position=False)

        summed_precision += precision
        nb_questions += 1

    mean_precision = summed_precision / nb_questions
    print(mean_precision)


print('hello')
