"""
Full process for the creation of the data to be injected into PiafAPI for service-public.fr
"""
from pathlib import Path

from src.data.knowledge_base import prepare_spf_kbase
from src.data.knowledge_base import arborescence
from src.data.clean_download_spf import clean_folder, download_and_save


path_to_xml = Path('./data/knowledge-base/xml_sp')
path_to_arbo_folder = Path('./data/knowledge-base')
path_to_json = Path('./data/knowledge-base/json_sp')
print('Running all script')

clean_folder(path_to_json)
clean_folder(path_to_xml)

path_to_xml.mkdir(parents=True, exist_ok=True)
path_to_json.mkdir(parents=True, exist_ok=True)

print('We will delete old data. and download new spf data')

download_and_save('https://lecomarquage.service-public.fr/vdd/3.0/part/zip/vosdroits-latest.zip', path_to_xml)
print('We successfully downloaded spf from data.gouv.fr')


print('We will now start the arbo creation now')
arborescence.main(path_to_xml, path_to_arbo_folder)
print('arbo finished')




print('...')
print('We will now start the creation of JSON files based on the XML')


prepare_spf_kbase.main(path_to_xml, path_to_json, path_to_arbo_folder / 'arborescence.json', n_jobs=1,
                       as_json=1, as_one=0)
