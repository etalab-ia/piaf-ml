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