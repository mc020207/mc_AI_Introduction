import numpy as np


def build(sort_labels, word2id, sentences, tag_list):
    A = np.zeros([len(sort_labels), len(sort_labels)])
    B = np.zeros([len(word2id), len(sort_labels)])
    Pi = np.zeros([len(sort_labels)])
    for tags in tag_list:
        Pi[tags[0]] += 1
        for idx in range(len(tags) - 1):
            A[tags[idx]][tags[idx + 1]] += 1
    for idx in range(len(sentences)):
        sentence = sentences[idx]
        tags = tag_list[idx]
        for idx2 in range(len(tags)):
            B[word2id[sentence[idx2]]][tags[idx2]] += 1
    A[A == 0] += 5e-2
    B[B == 0] += 5e-2
    Pi[Pi == 0] += 5e-2
    A = A / np.sum(A, axis=1, keepdims=True)
    B = B / np.sum(B, axis=0, keepdims=True)
    Pi = Pi / np.sum(Pi)
    A = np.log2(A)
    B = np.log2(B)
    Pi = np.log2(Pi)
    return A, B, Pi


def viterbi(word2id, A, B, Pi, sentence):
    sentence_len = len(sentence)
    tag_num = len(Pi)
    dp = np.zeros((tag_num, sentence_len))
    pre = np.zeros((tag_num, sentence_len))
    start = word2id.get(sentence[0])
    if start is None:
        now = np.ones(tag_num)
        now = now * np.log(1.0 / tag_num)
    else:
        now = B[start]
    dp[:, 0] = Pi + now
    pre[:, 0] = -1
    for idx in range(1, sentence_len):
        wordid = word2id.get(sentence[idx], None)
        if wordid is None:
            now = np.ones(tag_num)
            now = now * np.log(1.0 / tag_num)
        else:
            now = B[wordid]
        dp[:, idx] = [np.max(dp[:, idx - 1] + A[:, tag_id], 0) for tag_id in range(tag_num)] + now
        pre[:, idx] = [np.argmax(dp[:, idx - 1] + A[:, tag_id], 0) for tag_id in range(tag_num)]
    p = int(np.argmax(a=dp[:, sentence_len - 1], axis=0))
    path = [p]
    for idx in range(sentence_len - 1, 0, -1):
        p = int(pre[p, idx])
        path.append(p)
    path.reverse()
    return path
