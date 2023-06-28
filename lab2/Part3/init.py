import numpy as np

sorted_labels_eng = ["<PAD>", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

sorted_labels_chn = [
    "<PAD>", 'O',
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
    if language == "English":
        sort_labels = sorted_labels_eng
    else:
        sort_labels = sorted_labels_chn
    tag2id = {}
    for s in sort_labels:
        tag2id[s] = len(tag2id)
    f = open('./NER/' + language + '/' + mode + '.txt', 'r', encoding='utf-8')
    word2id = {}
    data_list = []
    sentence = []
    tags = []

    for i in range(5000000):
        s = f.readline()
        if s == '':
            break
        s = s[:-1]
        if s != '':
            word, tag = s.split(' ')
            if word2id.get(word) is None:
                word2id[word] = len(word2id)
            sentence.append(word)
            if tag2id.get(tag) is None:
                print(tag)
            tags.append(tag)
        elif len(sentence) != 0:
            data_list.append((sentence.copy(), tags.copy()))
            sentence.clear()
            tags.clear()
    if len(sentence) != 0:
        data_list.append((sentence.copy(), tags.copy()))
    return sort_labels, tag2id, word2id, data_list
