# -*- coding: utf-8 -*-


import mxnet as mx
import numpy as np
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    #1st block
    net.add(nn.Conv2D(channels=20,kernel_size=5))
    net.add(nn.BatchNorm(axis=1))  #默认X是4维，针对channel（默认channel的xias=1）进行batch normalization。与X.mean(axis=(0,2,3))是一样的。
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2,strides=2))
    #2nd block
    net.add(nn.Conv2D(channels=50,kernel_size=3))
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2,strides=2))
    net.add(nn.Flatten())
    #1st Dense
    net.add(nn.Dense(128))
    # net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    #2nd Dense
    net.add(nn.Dense(10))

import sys
sys.path.append('..')
from mxnet import autograd
import utils
from mxnet import nd
from mxnet import gluon



ctx = utils.try_gpu()
net.initialize(ctx=ctx)

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':1})



for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print "%d epoch: the train loss is %f, the train accuracy is %f, the test accuracy is %f."%\
          (epoch, train_loss / len(train_data),
         train_acc / len(train_data), test_acc)










