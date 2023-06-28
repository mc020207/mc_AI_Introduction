import torch.nn.functional as F
import torch.utils.data as torchio
import net
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision
from MyDataset import MyDataset

test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.Resize(224),
                          download=True)

dev_loader = torchio.DataLoader(MyDataset(test_data), batch_size=32)
idx = 5
model = torchvision.models.resnet18()
loss_fn = F.cross_entropy
metric = net.Accuracy(is_logist=True)
runner = net.Runner(model, None, loss_fn, metric)
runner.load_model('best_model' + str(idx) + '.pdparams')
score = runner.evaluate(dev_loader)
print(idx, ' ', score)
