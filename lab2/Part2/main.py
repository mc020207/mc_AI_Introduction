from sklearn_crfsuite import CRF
from init import init
from net import extract_labels, extract_features
from NER.check import check
from tqdm import tqdm
import pickle

language = 'English'
train_data = init(language, 'train')
X_train, y_train = extract_features(train_data, language), extract_labels(train_data)
crf = CRF(algorithm='ap', max_iterations=300, all_possible_transitions=True, verbose=True)
crf.fit(X_train, y_train)
savefile = open(language + '.param', 'wb')
pickle.dump(crf, savefile)
