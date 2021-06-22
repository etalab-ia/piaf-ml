# url = 'https://lecomarquage.service-public.fr/vdd/3.0/part/zip/vosdroits-latest.zip'

# we begin by deleting the previous folder if it exist
import shutil


def clean_folder(path):
    shutil.rmtree(path, ignore_errors=True)


import io
import zipfile

# we download the latest dataset and we save it
import requests


def download_and_save(url, path):
    r = requests.get(url)
    print("Is the service-public zip file from data.gouv.fr ok ?", r.ok)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
