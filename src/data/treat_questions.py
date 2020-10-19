import spacy
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import base64

print('loading nlp ...')
nlp = spacy.load("fr_core_news_sm")
print('loading completed !')

def remove_tags (line, tag_name):
    """
    :param line: the line from the xlm file
    :param tag_name: the name of the tag you want to remove
    :return: string without the xml tags
    """
    return line.replace(f'<{tag_name}>', '').replace(f'</{tag_name}>', '')

def process_incoming_message(incoming_message):
    """
    :param incoming_message:
    :return: a dict {'question': length of the question in words}
    Note that the punctuation if counted as a word
    """
    result = {}
    doc = nlp(incoming_message)
    list_sentences = list(doc.sents)
    for sentence in list_sentences:
        if str(sentence[-1]) == '?':
            result[sentence.text] = len(sentence)
    return result, len(list_sentences) == 1

xml_file = Path('../data/15082020-ServicePublic_QR_20170612_20180612.xml')
with open(xml_file) as filo:
    lines = [l.strip() for l in filo]

result = {}
new_line = {}
for l in tqdm(lines[:]):
    if "<reference>" in l:
        reference = remove_tags(l,'reference')
    if "<incoming>" in l:
        temp = remove_tags(l,'incoming')
        temp = base64.b64decode(temp).decode('utf-8')
        temp = temp.replace('\n','').replace('\\',"")
        new_line['incoming_message'] = temp
        questions, new_line['unique_question'] = process_incoming_message(temp)
    if "<outgoing>" in l:
        temp = remove_tags(l,'outgoing')
        temp = base64.b64decode(temp).decode('utf-8')
        temp = temp.replace('\n','').replace('\\',"")
        new_line['response'] = temp
    if "<url>" in l:
        temp = remove_tags(l, 'url')
        if 'url' not in new_line:
            new_line['url'] = temp
        elif 'url_2' not in new_line:
            new_line['url_2'] = temp
        elif 'url_3' not in new_line:
            new_line['url_3'] = temp
        else:
            new_line['url_4'] = temp #there are sometimes more links but they will be discarded
    if "</qr>" in l:
        for i, q in enumerate(questions):
            new_line_final = new_line.copy()
            new_line_final['question'] = q
            new_line_final['len_question'] = questions[q]
            result[reference + f'_{i}'] = new_line_final
        new_line = {}

df = pd.DataFrame.from_dict(result, orient='index')
df.to_csv(f'../data/{xml_file.stem}.csv')
print('Done ! Csv saved')