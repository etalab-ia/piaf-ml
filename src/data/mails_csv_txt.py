""""
This code converts all the .csv files "export*.csv" into a whole new mail.txt file
"""
import pandas as pd

mail = pd.read_csv('data/1_an_historique/mail.csv',sep=';',low_memory=False)
mail_list=mail['Corps du mail'].tolist()
a='.'.join(map(str,mail_list))
with open("data/1_an_historique/mail.txt", "w") as text_file:
    text_file.write(str(mail_list))