import hashlib
import subprocess
import socket
import os

from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

import mlflow


load_dotenv()

def add_extra_params(dict_params: dict):
    extra_parameters = {
        "date": datetime.today().strftime('%Y-%m-%d_%H-%M-%S'),
        "hostname": socket.gethostname()
    }

    dict_params.update(extra_parameters)
    experiment_id = hashlib.md5(str(dict_params).encode("utf-8")).hexdigest()[:4]
    dict_params.update({"experiment_id": experiment_id})


def create_run_ids(parameters_grid):
    """
    This function creates a list of ids for every run of the experiment to be launched.
    Each id is composed of the hash for the git commit of the code, the hash for the knowledge_base being used for testing,
    the hash for the params of the run.

    :param parameters_grid: the parameter grid created from the configuration file
    :return: a list of run ids
    """
    git_commit = subprocess.check_output("git rev-parse --short HEAD", encoding="utf-8").strip()
    hash_file = {}
    run_ids = []
    for param in parameters_grid:
        id = []
        file = param['squad_dataset']
        if file not in hash_file.keys():
            with open(file, 'r', encoding="utf-8") as f:
                file_content = f.read()
            file_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()[:8]
            hash_file[file] = file_hash
        hash_param  = hashlib.md5(str(param).encode("utf-8")).hexdigest()[:8]
        id = git_commit + hash_file[file] + hash_param
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
