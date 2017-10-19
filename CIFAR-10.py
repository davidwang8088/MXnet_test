# -*- coding: utf-8 -*-


# 训练model or 测试
is_training = True
# 使用demo数据集 or Kaggle数据集
demo = True

# if demo:
#     import zipfile
#     for fin in ['train_tiny.zip', 'test_tiny.zip', 'trainLabels.csv.zip']:
#         with zipfile.ZipFile('data/kaggle_cifar10/' + fin, 'r') as zin:
#             zin.extractall('data/kaggle_cifar10/')


import os
import shutil


#
def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))



##################################################
###     对训练数据进行分类    #######################
###     1.对60000个样本的label文件，构造dict{idx：label}
###     2.根据valid_ratio,并根据label的类别，分别拷贝至train，valid和test文件夹。
##################################################
def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    # 读取训练数据的label
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过表头
        lines = f.readlines()[1:]
        # 去掉l末尾空格+分割
        tokens = [l.rstrip().split(',') for l in lines]
        # 根据idx和label构造字典
        idx_label = dict(((int(idx), label) for idx, label in tokens))
    # 找到idx_label中所有的label值，去重
    labels = set(idx_label.values())

    # 得到训练样本的数量，os.listdir返回该目录下所有的文件和文件夹
    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    num_train_tuning = int (num_train * (1 - valid_ratio))
    assert num_train_tuning < num_train
    # 每个label中存储的训练图片数量
    num_train_tuning_per_label = num_train_tuning // len(labels)
    label_count = dict()


    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = int(train_file.split('.')[0])
        # 根据索引找到图片对应的label
        label = idx_label[idx]
        # 构造训练集
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        # train_file 放入‘train_valid’ 目录
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        # 如果label不存在，
        if label not in label_count or label_count[label] < num_train_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            #返回label的键值，如果第一次为空，则返回0。主要用来计数的dict
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))

    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))

if demo:
    # 注意：此处使用小训练集为便于网页编译。Kaggle的完整数据集应包括5万训练样本。
    train_dir = 'train_tiny'
    # 注意：此处使用小测试集为便于网页编译。Kaggle的完整数据集应包括30万测试样本。
    test_dir = 'test_tiny'
    # 注意：此处相应使用小批量。对Kaggle的完整数据集可设较大的整数，例如128。
    batch_size = 1
else:
    train_dir = 'train'
    test_dir = 'test'
    batch_size = 128

data_dir = 'data/kaggle_cifar10'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
valid_ratio = 0.1
reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)


from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np

# 对训练数据进行图像增强的操作，
def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                                    rand_crop=False, rand_resize=False, rand_mirror=True,
                                    mean=np.array([0.4914, 0.4822, 0.4465]),
                                    std=np.array([0.2023, 0.1994, 0.2010]),
                                    brightness=0, contrast=0,
                                    saturation=0, hue=0,
                                    pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
        # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 测试时，无需对图像做标准化以外的增强数据处理。
def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))


input_str = data_dir + '/' + input_dir + '/'
# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1,
                                     transform=transform_train)
# valid数据也不做增强变换！！！！！
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1,
                                     transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid',
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1,
                                     transform=transform_test)


loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')




# models
# alexnet ->                 ResNet-18

##ResNet-18
from mxnet.gluon import nn
from mxnet import nd

class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            # same_shape 表示输入和输出的shape是否一样。因为out+x，shape必须一致。
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels,kernel_size=3, padding=1, strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels,kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)



class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            #block1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            #block2
            for _ in range(3):
                net.add(Residual(channels=32))
            #block3
            net.add(Residual(channels=64), same_shape=False)
            for _ in range(3):
                net.add(Residual(channels=32))
            #block4
            net.add(Residual(channels=128), same_shape=False)
            for _ in range(3):
                net.add(Residual(channels=128))
            #block5
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

        def hybrid_forward(self, F, x):
            out = x
            for i, b in enumerate(self.net):
                out = b(out)
                if self.verbose:
                    print ('Block %d output'%(i+1, out.shape))
            return out

def get_net(ctx):
    num_outputs = 10
    net = ResNet(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier)
    return net


import datetime
import utils
# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params, 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in num_epochs:
        train_loss = 0.
        train_acc = 0.
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_training_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with gluon.autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print (epoch_str + time_str + ', lr' + str(trainer.learning_rate))



ctx = utils.try_gpu()
num_epochs = 1
learning_rate = 0.1
weight_decay = 5e-4
lr_period = 80
lr_decay = 0.1
net = get_net(ctx)
net.hybridize()

if is_training:
    train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)
else:
    import numpy as np
    import pandas as pd
    train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)

preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))

    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
    df.to_csv('submission.csv', index=False)

















