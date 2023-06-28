import numpy as np

sorted_labels_eng = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

sorted_labels_chn = [
    'O',
    'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
    , 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
    , 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
    , 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
    , 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
    , 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
    , 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
    , 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]


def init(language='English', mode='train'):
    f = open('./NER/' + language + '/' + mode + '.txt', 'r', encoding='utf-8')
    sentences = []
    sentence = []
    for i in range(10000000):
        s = f.readline()
        if s == '':
            break
        s = s[:-1]
        if s != '':
            word, tag = s.split(' ')
            sentence.append((word, tag))
        elif len(sentence) != 0:
            sentences.append(sentence.copy())
            sentence.clear()
    if len(sentence) != 0:
        sentences.append(sentence.copy())
    return sentences
