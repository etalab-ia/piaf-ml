import hashlib
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def add_extra_params(dict_params: dict):
    extra_parameters = {
        "date": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
        "hostname": socket.gethostname(),
    }

    dict_params.update(extra_parameters)
    experiment_id = hashlib.md5(str(dict_params).encode("utf-8")).hexdigest()[:4]
    dict_params.update({"experiment_id": experiment_id})


def hash_piaf_code():
    """
    This function lists all files in folders data and src at the exception of the files for experiment configuration.
    Then it returns the hash of the list of the md5 sums of the files :return: str with the hash of the list of the md5
    sums of the files
    """
    folders_to_hash = ["src"]
    list_files_to_hash = []
    for folder in folders_to_hash:
        for path in Path(folder).rglob("*.py"):
            if "_config.py" in path.name:
                continue
            list_files_to_hash.append(path)

    list_hash = []
    for file in list_files_to_hash:
        with open(file, "r", encoding="utf-8") as f:
            file_content = f.read()
        file_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()[:8]
        list_hash.append(file_hash)

    return hashlib.md5(str(list_hash).encode("utf-8")).hexdigest()[:8]


def get_list_past_run(client, experiment_name):
    """
    This function returns the list of the past experiments 
    :param client: the mlflow client 
    :param experiment_name: the name of the experiment 
    :return: the list of the past experiments
    """
    try:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        # create a dict with {run_name: run_id}
        list_past_run_names = {
            client.get_run(run.run_id).data.tags["mlflow.runName"]: run.run_id
            for run in client.list_run_infos(experiment_id)
            if run.status == "FINISHED"
        }
    except:
        list_past_run_names = {}

    return list_past_run_names


def create_run_ids(parameters_grid):
    """
    This function creates a list of ids for every run of the experiment to be launched. Each id is composed of the hash
    for the git commit of the code, the hash for the knowledge_base being used for testing, the hash for the params of
    the run.

    :param parameters_grid: the parameter grid created from the configuration file 
    :return: the run_ids: a list of run ids
    """
    git_commit = subprocess.check_output(
        "git rev-parse --short HEAD", encoding="utf-8", shell=True
    ).strip()
    hash_librairies = hashlib.md5(
        subprocess.check_output("pip freeze", encoding="utf-8", shell=True).encode(
            "utf-8"
        )
    ).hexdigest()[:8]
    hash_code = hash_piaf_code()
    hash_file = {}
    run_ids = []
    for param in parameters_grid:
        file = param["squad_dataset"]
        if file not in hash_file.keys():
            with open(file, "r", encoding="utf-8") as f:
                file_content = f.read()
            file_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()[:8]
            hash_file[file] = file_hash
        hash_param = hashlib.md5(str(param).encode("utf-8")).hexdigest()[:8]
        id = git_commit + hash_code + hash_librairies + hash_file[file] + hash_param
        run_ids.append(id)
    return run_ids


def prepare_mlflow_server():
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_SERVER_URI")

        if tracking_uri:
            # using databricks
            if "/Users/pavel.soriano" in tracking_uri:
                mlflow.set_tracking_uri("databricks")
                mlflow.set_experiment(tracking_uri)
                tqdm.write(f"MLflow tracking to databricks {tracking_uri}")
            # use remote server
            else:
                mlflow.set_tracking_uri(tracking_uri)
                tqdm.write(f"MLflow tracking to server {tracking_uri}")

                pass
        else:
            tqdm.write(f"MLflow tracking to local mlruns folder")
    except Exception as e:
        tqdm.write(f"Not using remote tracking servers. Error {e}")
        tqdm.write(f"MLflow tracking to local mlruns folder")
