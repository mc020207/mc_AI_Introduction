import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from tqdm import tqdm


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=self.stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果残差模块最后的输入输出格式不同则需要将他们化成相同的格式
        if in_channels != out_channels or stride != 1:
            self.use_1x1conv = True
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=self.stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.use_1x1conv = False

    def forward(self, inputs):
        y = F.relu(self.bn1(self.conv1(inputs)))
        y = self.bn2(self.conv2(y))
        if self.use_1x1conv:
            shortcut = self.shortcut(inputs)
            shortcut = self.bn3(shortcut)
        else:
            shortcut = inputs
        y = torch.add(shortcut, y)
        out = F.relu(y)
        return out


def make_first_module(in_channels):
    m1 = nn.Sequential(nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return m1


def resnet_module(input_channels, out_channels, num_res_blocks, stride=1):
    blk = []
    for i in range(num_res_blocks):
        if i == 0:
            blk.append(ResBlock(input_channels, out_channels,
                                stride=stride))
        else:
            blk.append(ResBlock(out_channels, out_channels))
    return blk


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18, self).__init__()
        m1 = make_first_module(in_channels)
        m2 = nn.Sequential(*resnet_module(64, 64, 2, stride=1))
        m3 = nn.Sequential(*resnet_module(64, 128, 2, stride=2))
        m4 = nn.Sequential(*resnet_module(128, 256, 2, stride=2))
        m5 = nn.Sequential(*resnet_module(256, 512, 2, stride=2))
        self.net = nn.Sequential(m1, m2, m3, m4, m5,
                                 # 汇聚层、全连接层
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.net(x)


class Runner(object):
    def __init__(self, model, optimizer, loss_fn, metric):
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
