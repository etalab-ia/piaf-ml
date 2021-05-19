"""
" In order to fine-tune a cammebert model with the domain of service-public.fr we need text. The 25k mails from SPF
seems like a good idea. We extract all the emails into a single text file corpus_mail_spf.txt This code converts all the
.csv files "export*.csv" into a whole new mail.txt file
"""
import sys

import pandas as pd
from tqdm import tqdm

if len(sys.argv) < 2:
    print(
        "Please give a path with the mails csv file (a single csv with all the mails' dump)"
    )
    exit()
mail_path = sys.argv[1]

mail = pd.read_csv(mail_path, sep=";", low_memory=False)
mail_list = mail["Corps du mail"].tolist()
mail_list = list(map(str, mail_list))
with open("data/corpus_mail_spf.txt", "w") as text_file:
    for line in tqdm(mail_list):
        text_file.write(line)
        text_file.write("\n")
