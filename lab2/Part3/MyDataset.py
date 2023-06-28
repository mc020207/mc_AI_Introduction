import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, examples, word_to_ix, tag_to_ix, unk_id):
        super(MyDataset, self).__init__()
        # 词典，用于将单词转为字典索引的数字
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unk_id = unk_id
        # 加载后的数据集
        self.examples = self.words_to_id(examples)

    def words_to_id(self, examples):
        tmp_examples = []
        for idx, example in enumerate(examples):
            seq, label = example
            # 将单词映射为字典索引的ID， 对于词典中没有的单词用[UNK]对应的ID进行替代
            seq = [self.word_to_ix.get(word, self.unk_id) for word in seq]
            label = [self.tag_to_ix.get(tag) for tag in label]
            tmp_examples.append([seq, label])
        return tmp_examples

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        return seq, label

    def __len__(self):
        return len(self.examples)


def collate_fn(batch_data, pad_tag):
    seqs, seq_lens, labels = [], [], []
    max_len = 0
    for example in batch_data:
        seq, label = example
        # 对数据截断并保存于seqs中7 567uyh
        seqs.append(seq)
        seq_lens.append(len(seq))
        labels.append(label)
        # 保存序列最大长度
        max_len = max(max_len, len(seq))
    # 对数据序列进行填充至最大长度
    for i in range(len(seqs)):
        seqs[i] = seqs[i] + [0] * (max_len - len(seqs[i]))
        labels[i] = labels[i] + [pad_tag] * (max_len - len(labels[i]))

    return (torch.tensor(seqs), torch.tensor(seq_lens, dtype=torch.int)), torch.tensor(labels)
