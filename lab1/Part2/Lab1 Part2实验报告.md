<center><big><big><big><big><big><big>Lab1 Part2实验报告</big></big></big></big></big></big></center>

<center><big><big><big>马成 20307130112</big></big></big></center>

## 代码结构

### 模型

这次实验我尝试使用了两种模型，一种是LeNet模型，一种是ResNet18模型。

#### LeNet

根据书上对LeNet的介绍直接搭建Lenet的网络结构，但是需要注意的是要在最后通过降维来方便后续全连接层的操作。

1. 用$in\_channel \times6$的大小为$5 \times5$的卷积核，输出6channel的$28 \times28$大小的图；

2. 用$2 \times2$，步长为2的最大池化层后，输出6channel的$14 \times14$的图；

3. 用$6\times 16$的大小为$5 \times5$的卷积核，得到16channel$10 \times10$大小的图；

4. 用$2 \times2$，步长为2的平均池化层后，输出16channel的$5 \times5$的图；

5. 用$16\times 120$的大小为$5 \times5$的卷积核，得到120channel$1 \times1$大小的图；

6. 此时，将特征图展平成1维，则有120个像素点，经过输入神经元个数为120，输出神经元个数为84的全连接层后，输出的长度变为84。

7. 再经过一个全连接层的计算，最终得到了长度为类别数的输出结果。

```python
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
```

#### ResNet18

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

### Dataset

​		所有原先的图片都是28\*28大小的矩阵，但是LeNet需要的是1\*32\*32，ResNet18需要的是1\*224\*224的矩阵，所以可以先在Dataset先处理。

```python
class MyDataset(torchio.Dataset):
    def __init__(self, dataset, transforms):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[0][idx], self.dataset[1][idx]
        image, label = np.array(image).astype('float32'), int(label)
        image = np.reshape(image, [28, 28])
        image = Image.fromarray(image.astype('uint8'), mode='L')
        image = self.transforms(image)
        image = np.array(image, dtype=np.float32)
        image = np.array([image], dtype=np.float32)
        return image, label

    def __len__(self):
        return len(self.dataset[0])
```

### Runner

​		Runner主要是用来管理train和未来测试的过程。

```python
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
                        f"[Evaluate] best accuracy performence has been updated: 
                        {self.best_score:.5f} --> {dev_score:.5f}")
                    self.best_score = dev_score
            global_step += 1
        if log_steps and epoch % log_steps == 0:
            print(
                f"[Train] epoch: {epoch}/{num_epochs}, step: 
                {global_step}/{num_training_steps}, loss: {loss.item():.5f}")
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
```



## 设计实验改进网络并论证

​		一开始我使用的是LeNet训练，最后在测试集上大致只有96%的正确率。因此我选择更加先进的ResNet18网络尝试。这个网络相比LeNet有几个特点，第一个特点是ResNet迭代了更多的层。如果单纯的只有这个特点按道理训练的难度会增加。但是实际上并没有，ResNet一般在第一个epoch就可以在验证集上达到十分优秀的正确率。

### 残差模块

​		第一个就是ResNet加入了残差模块·。假设我们希望用神经网络近似一个函数$h(x)$，可以把函数拆分成两个部分$h(x)=x+(h(x)-x)$，虽然可以证明神经网络对于$h(x)和(h(x)-x)$这两个函数都可以完成比较好的拟合，但是在实际的训练中可以发现后者更加容易被神经网络学习。因此在网络适当的加入残差模块可以更好的拟合也可以更快的收敛。

### 逐层归一化

​		第二点就是ResNet使用了BatchNorm2d这个模块，这个模块是用来操作逐层归一化的，通过对深层神经网络的某些层得到的数据做一个归一化使得网络更容易训练。原因大致有以下两点：

- 更好的尺度不变形：当训练深层神经网络的时候由于高层的神经网络的输入是由其下面若干层的迭代输出得到的，因此这个时候输入的分布会发生较大的变化，随着梯度下降的进行每一次参数更新会导致输入分布慢慢的积累变化最后和原分布相差较大。如果某个层的输入分布发生较大改变，这个层就要经历类似重新训练的过程。这个时候对某些层使用一个归一化可以较好的控制输入分布的变化，使得高层的输入相对稳定。
- 更平滑的优化地形：逐层归一化一方面可以使得大部分神经层的输入处于不饱和区域，从而让梯度变大，避免梯度消失问题；另一方面还可以使得神经网络的优化地形更加平滑，以及使梯度变得更加稳定，从而允许我们使用更大的学习率，并提高收敛速度。（以下只是我的个人猜测，感觉这里使用归一化可以获得优化地形的原因是机器学习算法应该是根据参数和模型的特征对在某一个区域内的数据较为敏感方便调整。那么如果通过归一化就可以控制这个输入到达这个算法敏感的部分，因此可以获得更好的优化地形）

​		在ResNet中使用的事逐层归一化方法中的批量归一化方法。令第𝑙 层的净输入为𝒛^(𝑙)^，神经元的输出为𝒂^(𝑙)^，即𝒂^(𝑙)^ = 𝑓(𝒛^(𝑙)^) = 𝑓(𝑾𝒂^(𝑙−1)^ + 𝒃)，此时根据经验我们一般将归一化操作作用在𝒛^(𝑙)^上。接下来就可以进行归一化操作，并且引入两个参数用于衡量缩放$\gamma$和平移参数$\beta$，$\hat Z^{(l)}=\frac{Z^{(l)-\mu}}{\sqrt{\sigma^2+\epsilon}}\gamma+\beta$其中$\mu$是当前输入batch的均值，$\sigma^2$是方差。特别的可以看到在诡异话的过程中已经有平移操作，所以在归一化前的线性层不需要添加bias

### Adam算法调整参数

​		在标准的梯度下降法中，每个参数在每次迭代时都使用相同的学习率．由于每个参数的维度上收敛速度都不相同，因此根据不同参数的收敛情况分别设置学习率。RMSprop算法就是这样的一种自适应学习率的方法。他通过每一次叠加梯度以及对总数进行比例衰减得到参数调整较为合适的速度。令$G_t=\beta G_{t-1}+(1-\beta)|g_t|^2$其中g是对应的梯度。参数更新的方式是$\Delta\Theta_t=-\frac{\alpha}{\sqrt{G_t+\epsilon}}g_t$

​		除了调整学习率之外，还可以进行梯度估计的修正，机梯度下降方法中每次迭代的梯度估计和整个训练集上的最优梯度并不一致，具有一定的随机性． 一种有效地缓解梯度估计随机性的方式是通过使用最近一段时间内的平均梯度来代替当前时刻的随机梯度来作为参数更新的方向，从而提高优化速度。动量法就是一个利用这个特点做出优化的方法，利用之前累计的动量来调整梯度。$\Delta\Theta_t=\rho\Delta\Theta_{t-1}-\alpha g_t$。

​		Adam算法可以看作动量法和RMSprop 算法的结合，不但使用动量作为参数更新方向，而且可以自适应调整学习率。具体的调整方法如下：
$$
M_t=\beta_1M_{t-1}+(1-\beta_1)g_t \:\:，\:\: G_t=\beta_2G_{t-1}+(1-\beta_2)|g_t|^2 \\
\hat M_t=\frac{M_t}{1-\beta_1^t} \:\:，\:\: \hat G_t=\frac{G_t}{1-\beta_2^t} \\
\Delta\Theta_t=-\frac{\alpha}{\sqrt{\hat G_t+\epsilon}}\hat M_t
$$

## 对网络设计的理解

​		在这个实验中我使用的都是卷积神经网络，和全连接神经网络相比，卷积神经网络的关注重点比较集中，注重提取局部的信息。但是全连接神经网络更加关注全局的特征，以为所有神经元都是和每一个上一层的神经元相连的。对于图像分类，我们可能更多的话关注局部的特征，因此符合这个条件，并且参数量相对较小的卷积神经网络可以训练的比全连接神经网络更好。

​		对于卷积核较小的层会关注比较小的局部特征，卷积核较大的层会关注较大的局部特征，但是随着网络的迭代，可能某一层中关注了上一层的若干范围，但是上一层的一个参数就关注了原输入的较大范围，因此随着网络的加深神经网络可以较好的关注到图片的各种范围的特征。同时在卷积神经网络中channel我认为就对应这图片的信息提取，有多少个channel就相当于提取多少种局部信息。

​		通过这次的学习我了解到了很多关于训练神经网络的技巧和理论，比如可以使用残差连接，逐层归一化已经Adam算法调整参数。这些方法可以使得模型更加容易训练收敛，也可以提高模型的适应性。

### 正确率对比

1. 全连接神经网络：86%-88%
2. LeNet网络：94%-98%
3. ResNet18： 98%-99.2%
