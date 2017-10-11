# -*- coding: utf-8 -*-

from mxnet import nd
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.autograd as autograde
import sys
sys.path.append('..')
import utils
from utils import load_data_fashion_mnist

#环境
try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
ctx

#数据
batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)


learning_rate = 0.5

#模型
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20,kernel_size=5,activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2,strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2,strides=2))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256,activation='relu'))
    net.add(gluon.nn.Dense(64,activation='relu'))
    net.add(gluon.nn.Dense(10))
net.initialize()

cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':learning_rate})

for e in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograde.record():
            output = net(data)
            loss = cross_entropy_loss(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)

    test_acc = utils.evaluate_accuracy(test_data,net,ctx=ctx)

    print "%d epoach: the train loss is %f, the train accracy is %f, the test accuracy is %f" %(e, train_loss / len(train_data), train_acc / len(train_data), test_acc)
