import torch
import torch.optim as optim
from net import BiLSTM_CRF
from init import init
from tqdm import tqdm
from MyDataset import MyDataset, collate_fn
from functools import partial
from torch.utils.data import DataLoader
from Runner import Runner

language = 'English'
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
UNK_ID = 5
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
BATCH_SIZE = 16
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
sort_labels, tag_to_ix, word_to_ix, train_data = init(language, 'train')
_, _, _, test_data = init(language, 'validation')
tag_to_ix.update({START_TAG: len(tag_to_ix), STOP_TAG: len(tag_to_ix) + 1})
train_dataset = MyDataset(train_data, word_to_ix, tag_to_ix, UNK_ID)
test_dataset = MyDataset(test_data, word_to_ix, tag_to_ix, UNK_ID)
collate_fn = partial(collate_fn, pad_tag=tag_to_ix[PAD_TAG])
train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
for idx in range(8, 50):
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    runner = Runner(model, optimizer, model.neg_log_likelihood, len(tag_to_ix))
    runner.train(train_dataloader, test_dataloader, num_epochs=2, log_steps=20, eval_steps=50, change_thread=0.8,
                 eval_steps2=5, save_path=language + 'Rubbish_Model' + str(idx))
