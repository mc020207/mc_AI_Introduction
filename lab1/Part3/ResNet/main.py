import torchvision.models
import torch.nn.functional as F
import torch.nn as nn
import net
import torch.utils.data as torchio
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.optim as opt
from MyDataset import MyDataset

train_data = dataset.MNIST(root="mnist",
                           train=True,
                           transform=transforms.Resize(224),
                           download=True)
test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.Resize(224),
                          download=True)

train2_data = []
dev_data = []
for i in range(len(train_data) - 500):
    X = train_data[i][0]
    y = train_data[i][1]
    train2_data.append((X, y))

for i in range(len(train_data) - 500, len(train_data)):
    X = train_data[i][0]
    y = train_data[i][1]
    dev_data.append((X, y))


lr = 0.001
batch_size = 32
train_loader = torchio.DataLoader(MyDataset(train_data), batch_size=batch_size, shuffle=True)
dev_loader = torchio.DataLoader(MyDataset(dev_data), batch_size=batch_size)
for idx in range(3, 30):
    model = torchvision.models.resnet18(pretrained=True)
    optimizer = opt.Adam(lr=lr, params=model.parameters())
    loss_fn = F.cross_entropy
    metric = net.Accuracy(is_logist=True)
    runner = net.Runner(model, optimizer, loss_fn, metric)
    log_steps = 1
    eval_steps = 5
    runner.train(train_loader, dev_loader, num_epochs=1, log_steps=log_steps,
                 eval_steps=eval_steps, save_path="best_modelrubish" + str(idx) + ".pdparams")
