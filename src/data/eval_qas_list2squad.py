"""
Transforms a list of [fiche_id, question, answer] into standard SQuAD format for evaluation.

Usage:
    eval_qas_list2squad.py <qas_path> <context_path> <output_path>

Arguments:
    <qas_path>              A path where to find the list of [fiche_id, question, answer] to transform
                                (should use: '../../data/spf_qr_test_raw.json')
    <context_path>          A path where to find the list of [fiche_id, context] from which the answers were extracted
                                (should use: '../../data/questions_spf.json')
    <output_path>           A path where to store the created SQuAD-like file
                                (should use: '../../data/spf_qr_test.json')
"""

import os
import json
from argopt import argopt
from pathlib import Path


def format_as_squad(fiche_id, context, question, answer_start, answer_text):
    """
    Once all parameters are found, formats in SQuAD-like output.
    """
    res = {
        'title': fiche_id,
        'paragraphs': [
            {
                'context': context,
                'qas': [
                    {
                        'question': question,
                        'answers': [
                            {
                                'answer_start': answer_start,
                                'text': answer_text
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return res


def list2squad(qas_path, context_path, output_path):
    """
    Doing conversion very inefficiently by loading both QAS and context in memory, then iterating over both.
    Will be improved if needed.
    """

    missed = 0
    data = []

    with open(qas_path, encoding='utf-8') as f:
        qas = json.load(f)
    with open(context_path, encoding='utf-8') as f:
        fiches = json.load(f)

    for qa in qas:
        fiche_id, question, answer = qa
        for fiche in fiches:
            if fiche[0] == fiche_id:
                start = fiche[2].find(answer)
                if start == -1:
                    missed += 1
                else:
                    data.append(format_as_squad(fiche_id=fiche_id,
                                                context=fiche[2],
                                                question=question,
                                                answer_start=start,
                                                answer_text=answer))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'data': data}, f, indent=4, ensure_ascii=False)
    print(f"Transformation done. Could not parse {missed} out of {len(qas)} questions.")


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    qas_path = Path(parser.qas_path)
    context_path = Path(parser.context_path)
    output_path = Path(parser.output_path)
    list2squad(qas_path=qas_path, context_path=context_path, output_path=output_path)
