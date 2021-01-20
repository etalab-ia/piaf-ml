from pathlib import Path
import json
import pandas as pd


def prepare_test_set(squad_path: str):
    """
    Loads the squad dataset.
    :param corpus_path: Path of the file containing the squad dataset
    :return: List of questions with the questions given in the squad form + the context
    [
        {
            id: "Id du pair question-réponse"
            question: "Question"
            context: "Paragraphe de l'article"
            title: "Titre de l'article Wikipedia"
            answers:[
                {
                    "answer_start": "Position de la réponse"
                    "text": "Réponse"
                }
            ]
        }
    ]
    """

    with open(file, encoding='UTF-8') as f:
        fquad = json.load(f)

    list_questions = []
    list_articles = fquad['data']
    for article in list_articles:
        title = article['title']
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for question in paragraph['qas']:
                question['title'] = title
                question['context'] = context
                list_questions.append(question)
    return list_questions

def split_test_set (file):
    with open(file, encoding='UTF-8') as f:
        fquad = json.load(f)
    fquad['data'] = [fquad['data'][0]]
    res_file = file.parent / (file.stem +'_fraction'+ file.suffix)

    with open(res_file, 'w', encoding='UTF-8') as f:
        json.dump(fquad,f)

def add_is_impossible (file):
    with open(file, encoding='UTF-8') as f:
        fquad = json.load(f)

    for question in fquad['data']:
        for paragraph in question['paragraphs']:
            for qas in paragraph['qas']:
                qas['is_impossible'] = False

    res_file = file.parent / (file.stem + '_with_impossible' + file.suffix)

    with open(res_file, 'w', encoding='UTF-8') as f:
        json.dump(fquad,f)



if __name__ == '__main__':
    file = Path('./test/samples/squad/small.json')
    add_is_impossible(file)
    prepare_knowledge_base()
