from init import init
from net import extract_labels, extract_features
from NER.check import check
from tqdm import tqdm
import pickle
mode='test'
language = "Chinese"
savefile = open(language + '.param', 'rb')
crf = pickle.load(savefile)
test_data = init(language, mode)
X_test, y_test = extract_features(test_data, language), extract_labels(test_data)
y_pred = crf.predict(X_test)
mypath = './NER/example_data/example_my_result.txt'
f = open(mypath, 'w', encoding='utf-8')
for sentence_data, my_tag_list in tqdm(zip(test_data, y_pred)):
    sentence = [x[0] for x in sentence_data]
    for idx in range(len(sentence)):
        f.write(sentence[idx] + ' ' + my_tag_list[idx] + '\n')
    f.write('\n')
f.close()
check(language, 'NER/' + language + '/'+mode+'.txt', mypath)
