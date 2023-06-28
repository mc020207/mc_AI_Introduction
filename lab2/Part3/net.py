import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _get_lstm_features(self, sentence, sentence_len):
        embeds = self.word_embeds(sentence)
        embeds2 = nn.utils.rnn.pack_padded_sequence(embeds, sentence_len.to('cpu'), batch_first=True, enforce_sorted=False)
        lstm_out, self.hidden = self.lstm(embeds2)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, sentence_len, tags):
        batch_size = feats.shape[0]
        score = torch.zeros(1, batch_size, device=self.device)
        tags = torch.cat([(torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.int, device=self.device).repeat(batch_size, 1)), tags],
                         dim=1)
        for batch_id in range(batch_size):
            tag_next = tags[batch_id, 1:sentence_len[batch_id] + 1]
            tag_now = tags[batch_id, 0:sentence_len[batch_id]]
            score[0][batch_id] = torch.sum(self.transitions[tag_next, tag_now])
            score[0][batch_id] += torch.sum(feats[batch_id, range(sentence_len[batch_id]), tag_next])
            score[0][batch_id] += self.transitions[self.tag_to_ix[STOP_TAG], tags[batch_id][sentence_len[batch_id]]]
        return score

    def _forward_alg(self, feats):
        batch_size = feats.shape[0]
        seq_len = feats.shape[1]
        init_alphas = torch.full((batch_size, self.tagset_size), -10000., device=self.device)
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        previous = init_alphas
        for iii in range(seq_len):
            obs = feats[:, iii, :]
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = obs[:, next_tag].unsqueeze(1).repeat(1, self.tagset_size)
                trans_score = self.transitions[next_tag].unsqueeze(0).repeat(batch_size, 1)
                next_tag_var = previous + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1, -1))
            previous = torch.cat(alphas_t).t()
        terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0).repeat(batch_size, 1)
        scores = torch.logsumexp(terminal_var, dim=1).view(1, -1)
        return scores

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        previous = init_vvars
        for obs in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = previous + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id.item())
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id.item()]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, data, tags):
        sentence, sentence_len = data
        feats = self._get_lstm_features(sentence, sentence_len)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, sentence_len, tags)
        return forward_score.mean() - gold_score.mean()

    def forward(self, data):
        sentence, sentence_len = data
        batch_size = sentence.shape[0]
        lstm_feats = self._get_lstm_features(sentence, sentence_len)
        scores = []
        tag_seqs = []
        for batch_id in range(batch_size):
            score, tag_seq = self._viterbi_decode(lstm_feats[batch_id][:sentence_len[batch_id]])
            scores.append(score)
            tag_seqs.append(tag_seq)
        return tag_seqs
