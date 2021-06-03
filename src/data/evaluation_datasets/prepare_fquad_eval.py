from pathlib import Path
import json


def add_is_impossible(fquad):
    """
    this function adds the key 'is_impossible' to a fquad formated dictionnary in order to go from squad v1 format to squad
    v2 format
    """
    for question in fquad["data"]:
        for paragraph in question["paragraphs"]:
            for qas in paragraph["qas"]:
                qas["is_impossible"] = False
    return fquad


def remove_answers_from_kb(fquad):
    """
    this function is used to remove the answers from the fquad formated dictionnary in order to use them as a knowledge
    base
    """
    for article in fquad["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qa["answers"] = []
                qa["is_impossible"] = True
    return fquad


def merge(kb_fquad, test_fquad):
    """
    this function is used for merging two dataset together. One is the knowledge base dataset, without the answers, and
    the other is the dataset that will be used for evaluation of the haystack pipeline
    """
    modified_fquad = kb_fquad.copy()
    existing_titles = [article["title"] for article in kb_fquad["data"]]
    for article in test_fquad["data"]:
        if article["title"] in existing_titles:
            index_title = existing_titles.index(article["title"])
            existing_contexts = [
                paragraph["context"]
                for paragraph in kb_fquad["data"][index_title]["paragraphs"]
            ]
            for paragraph in article["paragraphs"]:
                if paragraph["context"] in existing_contexts:
                    index_paragraph = existing_contexts.index(paragraph["context"])
                    existing_questions = [
                        qas["question"]
                        for qas in kb_fquad["data"][index_title]["paragraphs"][
                            index_paragraph
                        ]["qas"]
                    ]
                    for qas in paragraph["qas"]:
                        if qas["question"] in existing_questions:
                            index_qas = existing_questions.index(qas["question"])
                            modified_fquad["data"][index_title]["paragraphs"][
                                index_paragraph
                            ]["qas"][index_qas]["answers"] = qas["answers"]
                            modified_fquad["data"][index_title]["paragraphs"][
                                index_paragraph
                            ]["qas"][index_qas]["is_impossible"] = False
                        else:
                            existing_qas = modified_fquad["data"][index_title][
                                "paragraphs"
                            ][index_paragraph]["qas"]
                            existing_qas.append(qas)
                            modified_fquad["data"][index_title]["paragraphs"][
                                index_paragraph
                            ]["qas"] = existing_qas
                else:
                    existing_paragraphs = modified_fquad["data"][index_title][
                        "paragraphs"
                    ]
                    existing_paragraphs.append(paragraph)
                    modified_fquad["data"][index_title][
                        "paragraphs"
                    ] = existing_paragraphs
        else:
            existing_data = modified_fquad["data"]
            existing_data.append(article)
            modified_fquad["data"] = existing_data

    return modified_fquad


def save_dataset(modified_fquad, name="fquad_eval"):
    """save the dataset in the /data/evaluation-datasets folder"""
    res_file = Path("./data/evaluation-datasets") / (name + ".json")
    with open(res_file, "w", encoding="UTF-8") as f:
        json.dump(modified_fquad, f)

def count(squad):
    context_count = 0
    question_count = 0
    answer_count = 0
    for article in squad['data']:
        for context in article['paragraphs']:
            context_count += 1
            for question in context['qas']:
                if question["is_impossible"] == False:
                    question_count += 1
                    answer_count += len(question['answers'])
                    if len(question['answers']) == 0:
                        print(question["question"])
    print(f'Nb context = {context_count}')
    print(f'Nb question = {question_count}')
    print(f'Nb answer = {answer_count}')


def main(file_kb_fquad, file_test_fquad, name="fquad_eval"):
    with open(file_kb_fquad, encoding="UTF-8") as f_kb:
        kb_fquad = json.load(f_kb)

    with open(file_test_fquad, encoding="UTF-8") as f_test:
        test_fquad = json.load(f_test)

    kb_fquad = add_is_impossible(kb_fquad)
    test_fquad = add_is_impossible(test_fquad)

    kb_fquad = remove_answers_from_kb(kb_fquad)
    modified_fquad = merge(kb_fquad, test_fquad)

    # todo: add a merging of the different paragraphs
    save_dataset(modified_fquad, name)


if __name__ == '__main__':
    file_kb_fquad = Path("./clients/dila/knowledge_base/squad.json")
    file_test_fquad = Path("./data/evaluation-datasets/piaf-annotations.json")

    with open(file_kb_fquad, encoding="UTF-8") as f_kb:
        kb_fquad = json.load(f_kb)

    count(kb_fquad)

    with open(file_test_fquad, encoding="UTF-8") as f_test:
        test_fquad = json.load(f_test)

    add_is_impossible(test_fquad)
    count(test_fquad)

    merged = merge(kb_fquad, test_fquad)

    count(merged)

    with open(file_kb_fquad, 'w', encoding="UTF-8") as f_kb:
        json.dump(merged, f_kb)
    print(1)
