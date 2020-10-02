""""
This code converts all the .csv files "export*.csv" into a whole new mail.txt file
"""
import pandas as pd
from tqdm import tqdm

mail = pd.read_csv('data/1_an_historique/mail.csv', sep=';', low_memory=False)
mail_list = mail['Corps du mail'].tolist()
mail_list = list(map(str, mail_list))
with open("data/1_an_historique/corpus_mail_spf.txt", "w") as text_file:
    for line in tqdm(mail_list):
        text_file.write(line)
        text_file.write("\n")
