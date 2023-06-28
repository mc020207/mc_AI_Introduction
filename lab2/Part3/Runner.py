import torch
from sklearn import metrics
from tqdm import tqdm


class Runner(object):
    def __init__(self, model, optimizer, loss_fn, tag_num, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.tag_num = tag_num
        self.best_score = 0
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)

    def train(self, train_loader, dev_loader, num_epochs, log_steps, eval_steps, eval_steps2, change_thread, save_path):
        self.model.train()
        num_training_steps = num_epochs * len(train_loader)
        global_step = 0
        self.best_score = self.evaluate(dev_loader)
        self.model.train()
        print('start')
        for epoch in range(num_epochs):
            total_loss = 0
            for data in train_loader:
                X, y = data
                X = (X[0].to(self.device), X[1].to(self.device))
                y = y.to(self.device)
                loss = self.loss_fn(X, y)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if global_step and (global_step % eval_steps == 0 or global_step == (num_training_steps - 1)):
                    dev_score = self.evaluate(dev_loader)
                    if dev_score > change_thread:
                        eval_steps = eval_steps2
                    self.model.train()
                    if dev_score > self.best_score:
                        self.save_model(save_path)
                        print(
                            f"[Evaluate] best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
                        self.best_score = dev_score
                global_step += 1
                if global_step and global_step % log_steps == 0:
                    print(
                        f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}")
        print("[Train] Training done!")

    @torch.no_grad()
    def evaluate(self, dev_loader):
        self.model.eval()
        my_tags = []
        real_tags = []
        for (sentence, sentence_len), dev_tags in dev_loader:
            sentence = sentence.to(self.device)
            sentence_len = sentence_len.to(self.device)
            now_tags = self.model((sentence, sentence_len))
            for tags in now_tags:
                my_tags += tags
            for tags, now_len in zip(dev_tags, sentence_len):
                real_tags += tags[:now_len].tolist()
        dev_score = metrics.f1_score(y_true=real_tags, y_pred=my_tags, labels=range(2, self.tag_num), average='micro')
        return dev_score

    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        logits = self.model(x)
        return logits

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        model_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
