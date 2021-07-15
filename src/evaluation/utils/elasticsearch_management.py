import logging
import platform
import subprocess
import time

from elasticsearch import Elasticsearch




def launch_ES(hostname = "localhost", port = "9200"):
    logging.info("Search for Elasticsearch ...")
    es = Elasticsearch([f"http://{hostname}:{port}/"], verify_certs=True)
    if not es.ping():
        logging.info(f"Elasticsearch not found at http://{hostname}:{port}!")
        logging.info("Starting Elasticsearch ...")
        if platform.system() == "Windows":
            status = subprocess.run(
                f'docker run -d -p {port}:{port} -e "discovery.type=single-node" elasticsearch:7.6.2'
            )
        else:
            status = subprocess.run(
                [
                    f'docker run -d -p {port}:{port} -e "discovery.type=single-node" elasticsearch:7.6.2'
                ],
                shell=True,
            )
        time.sleep(30)
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch.")
    else:
        logging.info(f"Elasticsearch found at http://{hostname}:{port}")


def delete_indices(hostname = "localhost", port = "9200", index="document"):
    logging.info(f"Delete index {index} inside Elasticsearch ...")
    es = Elasticsearch([f"http://{hostname}:{port}/"], verify_certs=True)
    es.indices.delete(index=index, ignore=[400, 404])


def prepare_mapping(mapping, title_boosting_factor=1, embedding_dimension=512):
    mapping["mappings"]["properties"]["name"]["boost"] = title_boosting_factor
    mapping["mappings"]["properties"]["emb"]["dims"] = embedding_dimension
