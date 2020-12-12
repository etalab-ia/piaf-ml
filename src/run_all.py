"""
Full process for the creation of the data to be injected into PiafAPI for service-public.fr
"""


print('Runing all script')

from data import clean_download_spf
print('We will delete old data. and download new spf data')
clean_download_spf.clean_folder('../data/vosdroits-latest')
clean_download_spf.download_and_save('https://lecomarquage.service-public.fr/vdd/3.0/part/zip/vosdroits-latest.zip',"../data/vosdroits-latest")
print('We successfully downloaded spf from data.gouv.fr')


print('We will now start the arbo creation now')
from data import arborescence
arborescence.main('../data/vosdroits-latest', '../results', 1)

print('arbo finished')

print('We will now start the creation of JSON files based on the XML')

clean_download_spf.clean_folder('../results/resultsxml2txt')

from pathlib import Path
Path("../results/resultsxml2txt").mkdir(parents=True, exist_ok=True)

from data import fiches_xml2txt
fiches_xml2txt.main('../data/vosdroits-latest', '../results/resultsxml2txt', '../results/arborescence.json', n_jobs=1, as_json=1, as_one=0)

print('arbo finished')