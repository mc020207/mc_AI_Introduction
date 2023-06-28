import torch.nn.functional as F
import init
import torch.utils.data as torchio
import net

trainSet, validateSet, testSet = init.init_project()
batch_size = 32
test_loader = torchio.DataLoader(testSet, batch_size=batch_size)
model = net.LeNet(in_channels=1, num_classes=12)
idx = 23
loss_fn = F.cross_entropy
metric = net.Accuracy()
runner = net.Runner(model, None, loss_fn, metric)
runner.load_model('best_model' + str(idx) + '.pdparams')
score = runner.evaluate(test_loader)
print(idx, ' ', score.item())
