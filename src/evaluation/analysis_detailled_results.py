import json
import pandas as pd
import matplotlib.pyplot as plt

notheme_sparse = json.load(open("./results/ea7a_detailed_results.json"))
notheme_dense = json.load(open("./results/4fd9_detailed_results.json"))

theme_sparse = json.load(open("./results/24bf_detailed_results.json"))
theme_dense = json.load(open("./results/3921_detailed_results.json"))

dossier_sparse = json.load(open("./results/d4ad_detailed_results.json"))
dossier_dense = json.load(open("./results/4278_detailed_results.json"))

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

    return pd.DataFrame(data, columns=['score_dense','score_sparse','result'])

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

    return pd.DataFrame(data, columns=['score_dense','score_sparse','result'])

def get_detailled_data_from_experience(dense_json, sparse_json):
    """
    This function takes two json and check the success and errors in common
    """
    data = []
    results = 'successes'
    for question in dense_json[results].keys():
        if question in list(sparse_json[results].keys()):
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

    return pd.DataFrame(data, columns=['score_dense','score_sparse','result'])

df_res = get_detailled_data_from_experience(notheme_dense, notheme_sparse)
# df_res = pd.concat([df_res, get_detailled_data_from_experience(theme_dense, theme_sparse)])
# df_res = pd.concat([df_res, get_detailled_data_from_experience(dossier_dense, dossier_sparse)])

for res in df_res.result.unique():
    x = df_res[df_res.result == res]['score_dense']
    y = df_res[df_res.result == res]['score_sparse']
    plt.scatter(x, y, label=res)
plt.legend()
plt.xlabel('score_dense')
plt.ylabel('score_sparse')
plt.show()



print('hello')
