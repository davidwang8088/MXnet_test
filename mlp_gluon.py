# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import utils

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

batch_size = 256
#load data
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

#model
net  = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    #net.add(gluon.nn.Dense(256, activation = "relu"))
    net.add(gluon.nn.Dense(10))
net.initialize()

#损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
#训练器
trainer = gluon.Trainer(net.collect_params(),'SGD',{'learning_rate': 0.5})


#训练
epoch = 5

for e in range(epoch):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)

    test_acc = utils.evaluate_accuracy(test_data,net)
    print '%d epoch:    the training loss is %f, the training accuracy is %f, the test accuracy is %f' %(e, train_loss / len(train_data), train_acc / len(train_data), test_acc)
