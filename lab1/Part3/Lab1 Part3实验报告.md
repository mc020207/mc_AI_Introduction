<center><big><big><big><big><big><big>Lab1 Part3实验报告</big></big></big></big></big></big></center>

<center><big><big><big>马成 20307130112</big></big></big></center>

## 使用的模型

本次我使用了ResNet18模型（由于我Lab2就写过了ResNet18模型了，这里引用一下Lab的报告）

#### ResBlock

​		首先编写ResBlock，ResNet方便训练的原因就是增加了残差网络。可以看到一个残差单元就是两个$3 \times 3$的卷积。残差网络希望拟合的是$f(x)-x$所有在最后输出的时候需要将x加回去这里就要求x和经过两层卷积神经网络后得到的张量大小完全一致，如果不能完全一致那么只能用$1\times1$的卷积将他们变成一样的，最后在forward中将他们加起来。

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, 
                               stride=self.stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果残差模块最后的输入输出格式不同则需要将他们化成相同的格式
        if in_channels != out_channels or stride != 1:
            self.use_1x1conv = True
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=self.stride, 
                                      bias=False)
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
```

### ResNet18整体结构

1. 包含了一个步长为2，大小为$7 \times 7$的卷积层，卷积层的输出通道数为64，卷积层的输出经过批量归一化、ReLU激活函数的处理后，接了一个步长为2的$3 \times 3$的最大汇聚层；

2. 两个残差单元，output_channels=64，图尺寸不变；
3. 两个残差单元，output_channels=128，图尺寸减半；
4. 两个残差单元，经过运算后，output_channels=256，图尺寸减半；
5. 两个残差单元，经过运算后，output_channels=512，图尺寸减半；
6. 包含了一个全局平均汇聚层，将特征图变为$1 \times 1$的大小，最终经过全连接层计算出最后的输出。

```python
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
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), 
                                 nn.Linear(512, num_classes))

    def forward(self, x):
        return self.net(x)
```

​		在torch中的默认的情况是ResNet18(in_channels=3, num_classes=1000)

### 对于预训练网络的微调

​		由于默认的模型是ResNet18(in_channels=3, num_classes=1000)，而数据集的channel是1，并且num_class是10，因此我在前面加入了一个1\*1的卷积神经网络，在后面加入了一个1000\*10的线性层

```python
model = torchvision.models.resnet18()
model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1), model,
                      nn.Linear(in_features=1000, out_features=10))
```

### “预训练+微调”的理解

​		预训练模型是在大数据集上训练得到的一个有较强泛化能力的模型。但是这个模型并不能不做修改的适用于所有的任务，我们需要对这个模型进行一些微调来完成特定的任务。比如在这个任务中，预训练模型的输入和输出张量的shape和我们希望的不一样，可以通过直接修改预训练模型的结构或者在外围修改网络的结构。

​		预训练模型之所以可以有比较好的作用是基于一个假设：源域中的数据与目标域中的数据可以共享一些模型的参数。一般情况下这个假设是可以成立的，比如在这次的训练任务中有同学提醒我可以通过某些层的参数来加快训练效率（我最后这样尝试）。

​		预训练+微调具有如下优势：不需要针对新任务从头开始训练网络，节省了时间成本；预训练好的模型通常都是在大数据集上进行的，无形中扩充了我们的训练数据，使得模型更鲁棒、泛化能力更好；微调实现简单，使我们只关注自己的任务即可。但是在一些工作中指出在相同的任务上，预训练模型与从头开始训练相比，大大缩短了训练时间且加快了训练的收敛速度。在结果的提升上，他们的结论是，预训练模型只会对最终的结果有着微小的提升。

​		预训练模型对于迁移任务的作用有：在大型数据集上，预训练的性能决定了下游迁移任务的下限，即预训练模型可以作为后续任务的基准模型；在细粒度任务上，预训练模型无法显著提高最终的结果；与随机初始化相比，当训练数据集显著增加时预训练带来的提升会越来越小。即当训练数据较少时预训练能够带来较为显著的性能提升。

​		在模型的鲁棒性方面，预训练模型可以在以下场景中提高模型的鲁棒性：

- 对于标签损坏的情况，即噪声数据，预训练模型可以提高最终结果的AUC；
- 对于类别不均衡任务，预训练模型提高了最终结果的准确性；
- 对于对抗扰动的情况，预训练模型可以提高最终结果的准确性；
- 对于不同分布的数据，预训练模型带来了巨大的效果提升；
- 对于校准任务，预训练模型同样能提升结果置信度。
