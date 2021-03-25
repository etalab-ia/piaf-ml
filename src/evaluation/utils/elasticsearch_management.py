import subprocess
import time
import platform
import logging

from elasticsearch import Elasticsearch

port = '9200'

def launch_ES():
    logging.info("Search for Elasticsearch ...")
    es = Elasticsearch([f'http://localhost:{port}/'], verify_certs=True)
    if not es.ping():
        logging.info("Elasticsearch not found !")
        logging.info("Starting Elasticsearch ...")
        if platform.system() == 'Windows':
            status = subprocess.run(
                f'docker run -d -p {port}:{port} -e "discovery.type=single-node" elasticsearch:7.6.2'
            )
        else:
            status = subprocess.run(
                [f'docker run -d -p {port}:{port} -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
            )
        time.sleep(30)
        if status.returncode:
            raise Exception(
                "Failed to launch Elasticsearch.")
    else:
        logging.info("Elasticsearch found !")

def delete_indices(index='document'):
    logging.info(f"Delete index {index} inside Elasticsearch ...")
    es = Elasticsearch([f'http://localhost:{port}/'], verify_certs=True)
    es.indices.delete(index=index, ignore=[400, 404])

def prepare_mapping (mapping, preprocessing, title_boosting_factor =1, embedding_dimension=512):
    mapping["mappings"]["properties"]["name"]["boost"] = title_boosting_factor
    mapping["mappings"]["properties"]["emb"]["dims"] = embedding_dimension
    if not preprocessing:
        mapping['settings'] = {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": 'standard',
                            }
                        }
                    }
                }
