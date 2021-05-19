"""
This file splits a text file containing the SPF fiches raw text into two files: train.txt and test.txt. To use it, pass
a percentage of the desired size for the train
"""
import os
import sys
# TODO: This should be included in mails_csv_txt
from pathlib import Path

print(os.getcwd())

if len(sys.argv) < 3:
    print(
        "Please indicate the path of the file to split and the train percentage. Exiting..."
    )
    exit()


path_file = Path(sys.argv[1])
TRAIN_PCT = float(sys.argv[2])

file_size = path_file.stat().st_size
train_size = int(file_size * TRAIN_PCT)
with open(path_file) as fiches:
    train_content = fiches.read(train_size)
    test_content = fiches.read()

with open("data/train_fiches.txt", "w") as out:
    out.write(train_content)
with open("data/test_fiches.txt", "w") as out:
    out.write(test_content)
