import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from tqdm import tqdm


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.linear6 = nn.Linear(in_features=120, out_features=84)
        self.linear7 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # 卷积部分
        output = F.relu(self.conv1(x))
        output = self.pool2(output)
        output = F.relu(self.conv3(output))
        output = self.pool4(output)
        output = F.relu(self.conv5(output))
        # 此处将Tensor降维方便后续全连接层的处理
        output = torch.squeeze(output, 2)
        output = torch.squeeze(output, 2)
        # 全连接部分
        output = F.relu(self.linear6(output))
        output = self.linear7(output)
        return output


class Runner(object):
    def __init__(self, model, optimizer, loss_fn, metric, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.best_score = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def train(self, train_loader, dev_loader, num_epochs, log_steps, eval_steps, save_path):
        self.model.train()
        num_training_steps = num_epochs * len(train_loader)
        global_step = 0
        self.best_score = self.evaluate(dev_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for data in train_loader:
                X, y = data
                X = X.to(self.device)
                y = y.to(self.device)
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if global_step % eval_steps == 0 or global_step == (num_training_steps - 1):
                    dev_score = self.evaluate(dev_loader)
                    self.model.train()
                    if dev_score > self.best_score:
                        self.save_model(save_path)
                        print(
                            f"[Evaluate] best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
                        self.best_score = dev_score
                global_step += 1
            if log_steps and epoch % log_steps == 0:
                print(
                    f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}, loss: {loss.item():.5f}")
        print("[Train] Training done!")

    @torch.no_grad()
    def evaluate(self, dev_loader):
        self.model.eval()
        self.metric.reset()
        for data in dev_loader:
            X, y = data
            X = X.to(self.device)
            y = y.to(self.device)
            logits = self.model(X)
            self.metric.update(logits, y)
        dev_score = self.metric.accumulate()
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


class Accuracy(Metric):
    def compute(self):
        pass

    def __iter__(self):
        pass

    def __init__(self):
        self.num_correct = 0
        self.num_count = 0

    def update(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        labels = torch.squeeze(labels, dim=-1)
        batch_correct = torch.sum(torch.eq(preds, labels))
        batch_count = len(labels)
        self.num_correct += batch_correct
        self.num_count += batch_count

    def accumulate(self):
        if self.num_count == 0:
            return 0
        return self.num_correct / self.num_count

    def reset(self):
        self.num_correct = 0
        self.num_count = 0
