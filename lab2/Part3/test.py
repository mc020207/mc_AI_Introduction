import torch
import torch.optim as optim
from net import BiLSTM_CRF
from init import init
from tqdm import tqdm
from MyDataset import MyDataset, collate_fn
from functools import partial
from torch.utils.data import DataLoader
from Runner import Runner
from NER.check import check
mode='test'
language = 'Chinese'
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
UNK_ID = 5
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
BATCH_SIZE = 32
sort_labels, tag_to_ix, word_to_ix, train_data = init(language, 'train')
_, _, _, test_data = init(language, mode)
tag_to_ix.update({START_TAG: len(tag_to_ix), STOP_TAG: len(tag_to_ix) + 1})
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
test_dataset = MyDataset(test_data, word_to_ix, tag_to_ix, UNK_ID)
collate_fn = partial(collate_fn, pad_tag=tag_to_ix[PAD_TAG])
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
runner = Runner(model, None, None, len(tag_to_ix))
idx=30 if language=='Chinese' else 11
model_state_dict = torch.load(language + "Best_Model"+str(idx))
model.load_state_dict(model_state_dict)
mypath = './NER/example_data/example_my_result.txt'
f = open(mypath, 'w', encoding='utf-8')
with torch.no_grad():
    batch_num = 0
    for data, tags in tqdm(test_dataloader):
        model.zero_grad()
        y = model(data)
        for batch_id in range(len(data[0])):
            for i in range(len(test_data[batch_num * BATCH_SIZE + batch_id][0])):
                f.write(test_data[batch_num * BATCH_SIZE + batch_id][0][i] + ' ' + sort_labels[y[batch_id][i]] + '\n')
            f.write('\n')
        batch_num += 1
f.close()
check(language, 'NER/' + language + '/'+mode+'.txt', mypath)
