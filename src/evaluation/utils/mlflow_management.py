from dotenv import load_dotenv
from tqdm import tqdm
import mlflow
import os

load_dotenv()


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
