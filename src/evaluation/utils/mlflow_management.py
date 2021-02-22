from dotenv import load_dotenv
from tqdm import tqdm
import mlflow
import os

load_dotenv()


def prepare_databricks():
    try:
        if os.getenv("DATABRICKS_EXPERIMENT"):
            mlflow.set_tracking_uri("databricks")
            mlflow.set_experiment(os.getenv("DATABRICKS_EXPERIMENT"))
            tqdm.write(f"MLflow tracking to databricks {os.getenv('DATABRICKS_EXPERIMENT')}")
        else:
            tqdm.write(f"MLflow tracking to local mlruns folder")
    except:
        tqdm.write(f"MLflow tracking to local mlruns folder")