import torch.nn.functional as F
import net
import torch.utils.data as torchio
import init
import torch.optim as opt

trainSet, validateSet, testSet = init.init_project()
lr = 0.001
batch_size = 32
train_loader = torchio.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
dev_loader = torchio.DataLoader(validateSet, batch_size=batch_size)
for idx in range(15, 50):
    model = net.ResNet18(in_channels=1, num_classes=12)
    optimizer = opt.Adam(lr=lr, params=model.parameters())
    loss_fn = F.cross_entropy
    metric = net.Accuracy()
    runner = net.Runner(model, optimizer, loss_fn, metric)
    log_steps = 1
    eval_steps = 15
    runner.train(train_loader, dev_loader, num_epochs=5, log_steps=log_steps,
                 eval_steps=eval_steps, save_path="best_model" + str(idx) + ".pdparams")
