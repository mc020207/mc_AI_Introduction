import torch.nn.functional as F
import net
import torch.utils.data as torchio
import init
import torch.optim as opt

trainSet, validateSet, testSet = init.init_project()
lr = 0.08
batch_size = 64
train_loader = torchio.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
dev_loader = torchio.DataLoader(validateSet, batch_size=batch_size)
model = net.LeNet(in_channels=1, num_classes=12)
optimizer = opt.Adam(lr=lr / 10, params=model.parameters())
loss_fn = F.cross_entropy
metric = net.Accuracy()
runner = net.Runner(model, optimizer, loss_fn, metric)
runner.train(train_loader, dev_loader, num_epochs=80, log_steps=10, eval_steps=15,
             save_path="best_modelrubbish.pdparams")
