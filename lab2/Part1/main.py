from init import init
from net import build, viterbi
from tqdm import tqdm
from NER.check import check

language = 'English'
mode='test'
sort_labels, word2id, train_sentences, train_tag_list = init(language, 'train')
A, B, Pi = build(sort_labels, word2id, train_sentences, train_tag_list)
_, _, validation_sentences, test_tag_list = init(language, mode)
mypath = './NER/example_data/example_my_result.txt'
f = open(mypath, 'w', encoding='utf-8')
for sentence in tqdm(validation_sentences):
    my_tag_list = viterbi(word2id, A, B, Pi, sentence)
    assert (len(sentence) == len(my_tag_list))
    for idx in range(len(sentence)):
        f.write(sentence[idx] + ' ' + sort_labels[my_tag_list[idx]] + '\n')
    f.write('\n')
f.close()
check(language, 'NER/' + language + '/'+mode+'.txt', mypath)
