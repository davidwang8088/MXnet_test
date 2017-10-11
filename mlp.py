# -*- coding: utf-8 -*-

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import numpy as np
import utils
import matplotlib.pyplot as plt


#数据
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)


num_input = 28*28
num_output = 10
num_hidden = 256

#标准差， 根据Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
weight_sacle = np.sqrt(2.0 / batch_size )
location = 0

W1 = nd.random_normal(shape = (num_input, num_hidden),scale = weight_sacle, loc = location)
b1 = nd.random_normal(shape = (num_hidden))
W2 = nd.random_normal(shape = (num_hidden,num_hidden),scale = weight_sacle, loc = location)
b2 = nd.random_normal(shape = (num_hidden))
W3 = nd.random_normal(shape = (num_hidden,num_output),scale = weight_sacle, loc = location)
b3 = nd.random_normal(shape = (num_output))

params = [W1,b1,W2,b2,W3,b3]
#开辟求导空间
for param in params:
    param.attach_grad()


def relu(x):
    return nd.maximum(x,0)

def net(X):
    hidden = nd.dot(X.reshape((-1,num_input)),W1) + b1
    hidden = relu(hidden)
    output = nd.dot(hidden,W2) + b2
    return output # 这里只是

softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


epochs = 5
learning_rate = 0.5

for e in range(epochs):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy_loss(output,label) # 借用gluon中的softmax_cross_entropy_loss
        loss.backward() # 如果loss不是一个标量，那么loss.backward()等价于nd.sum(loss).backward().*2
        SGD(params,learning_rate / batch_size )

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print '%d epoch: the TRAIN loss is %f, the TRAIN accuracy is %f, the TEST accuracy is %f .' %(e, train_loss / len(train_data), train_acc / len(train_data), test_acc)
