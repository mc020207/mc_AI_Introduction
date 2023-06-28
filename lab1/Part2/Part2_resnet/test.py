import torch.nn.functional as F
import init
import torch.utils.data as torchio
import net

trainSet, validateSet, testSet = init.init_project()
batch_size = 64
train_loader = torchio.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
dev_loader = torchio.DataLoader(validateSet, batch_size=batch_size)
test_loader = torchio.DataLoader(testSet, batch_size=batch_size)
model = net.ResNet18(in_channels=1, num_classes=12)
print(len(test_loader))
idx=31
loss_fn = F.cross_entropy
metric = net.Accuracy()
runner = net.Runner(model, None, loss_fn, metric)
runner.load_model('best_model' + str(idx) + '.pdparams')
score = runner.evaluate(test_loader)
print(idx, ' ', score.item())
