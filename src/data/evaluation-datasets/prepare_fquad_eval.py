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

    with open(squad_path, encoding='UTF-8') as f:
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

def add_is_impossible (fquad):
    # with open(file, encoding='UTF-8') as f:
    #     fquad = json.load(f)

    for question in fquad['data']:
        for paragraph in question['paragraphs']:
            for qas in paragraph['qas']:
                qas['is_impossible'] = False

    # res_file = file.parent / (file.stem + '_with_impossible' + file.suffix)
    #
    # with open(res_file, 'w', encoding='UTF-8') as f:
    #     json.dump(fquad,f)

    return fquad

def remove_answers_from_kb(fquad):
    for article in fquad['data']:
        for paragraph in article['paragraphs']:
            for qas in paragraph['qas']:
                for answer in qas['answers']:
                    answer['text'] = ""
    return fquad

def merge(kb_fquad, test_fquad):
    modified_fquad = kb_fquad.copy()
    existing_titles = [article['title'] for article in kb_fquad['data']]
    for article in test_fquad['data']:
        if article['title'] in existing_titles:
            index_title = existing_titles.index(article['title'])
            existing_contexts = [paragraph['context'] for paragraph in kb_fquad['data'][index_title]['paragraphs']]
            for paragraph in article['paragraphs']:
                if paragraph['context'] in existing_contexts:
                    index_paragraph = existing_contexts.index(paragraph['context'])
                    existing_questions = [qas['question'] for qas in kb_fquad['data'][index_title]['paragraphs'][index_paragraph]['qas']]
                    for qas in paragraph['qas']:
                        if qas['question'] in existing_questions:
                            index_qas = existing_questions.index(qas['question'])
                            for answer in qas['answers']:
                                existing_texts = [answer['text'] for answer in kb_fquad['data'][index_title]['paragraphs'][index_paragraph]['qas'][index_qas]['answers']]
                                if answer['text'] not in existing_texts:
                                    existing_answers = modified_fquad['data'][index_title]['paragraphs'][index_paragraph]['qas'][index_qas]['answers']
                                    existing_answers.append(answer)
                                    modified_fquad['data'][index_title]['paragraphs'][index_paragraph]['qas'][index_qas]['answers'] = existing_answers
                        else:
                            existing_qas = modified_fquad['data'][index_title]['paragraphs'][index_paragraph]['qas'][index_qas]
                            existing_qas.append(qas)
                            modified_fquad['data'][index_title]['paragraphs'][index_paragraph]['qas'][index_qas] = existing_qas
                else:
                    existing_paragraphs = modified_fquad['data'][index_title]['paragraphs']
                    existing_paragraphs.append(paragraph)
                    modified_fquad['data'][index_title]['paragraphs'] = existing_paragraphs
        else:
            existing_data = modified_fquad['data']
            existing_data.append(article)
            modified_fquad['data'] = existing_data

    return  modified_fquad

def save_dataset(modified_fquad, name='fquad_eval'):
    res_file = Path('./data/evaluation-datasets') / (name + '.json')
    with open(res_file, 'w', encoding='UTF-8') as f:
        json.dump(modified_fquad,f)



def main(file_kb_fquad,file_test_fquad):
    with open(file_kb_fquad, encoding='UTF-8') as f_kb:
        kb_fquad = json.load(f_kb)

    with open(file_test_fquad, encoding='UTF-8') as f_test:
        test_fquad = json.load(f_test)

    kb_fquad = add_is_impossible(kb_fquad)
    test_fquad = add_is_impossible(test_fquad)

    kb_fquad = remove_answers_from_kb(kb_fquad)
    modified_fquad = merge(kb_fquad, test_fquad)

    #todo: add a merging of the different paragraphs
    save_dataset(modified_fquad, name='fquad_eval')

if __name__ == '__main__':
    file_kb_fquad = Path('./data/evaluation-datasets/fquad_train.json')
    file_test_fquad = Path('./data/evaluation-datasets/fquad_valid.json')

    main(file_kb_fquad,file_test_fquad)

