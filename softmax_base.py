# -*- coding: utf-8 -*-

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import numpy as np

import utils

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

import matplotlib.pyplot as plt

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

##debug
# data, label = mnist_train[0:9]
# show_images(data)
# print(get_text_labels(label))


batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)


num_input = 784
num_output = 10

W = nd.random_normal(shape = (num_input, num_output))
b = nd.random_normal(shape = (num_output))
params = [W,b]

#开辟求导空间
for param in params:
    param.attach_grad()

def softmax(x):
    exp = nd.exp(x)
    sum_exp = nd.sum(exp,axis=1, keepdims=True)
    return exp / sum_exp

    # temp = nd.dot(x.reshape((-1,num_input)), W) + b
    # return nd.exp(temp)[label] / sum(nd.exp(temp))

def net(X):
    output = nd.dot(X.reshape((-1,num_input)),W) + b
    return softmax(output)

def cross_entropy_loss(yhat,y):
    # return -nd.log(nd.pick(yhat,y))
    return - nd.log(yhat.pick(y))

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def Cal_Acc(output, label):
    return nd.mean(nd.argmax(output,axis=1)==label).asscalar()

def evaluate_acc(data_iterator, net):
    total_acc = 0
    for data, label in data_iterator:
        output = net(data)
        total_acc += Cal_Acc(output,label)
    return total_acc / len(data_iterator)


epochs = 5
learning_rate = .001

for e in range(epochs):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy_loss(output,label) # 这里loss不求和
        loss.backward() # 如果loss不是一个标量，那么loss.backward()等价于nd.sum(loss).backward().*2
        SGD(params,learning_rate)

        train_loss += nd.mean(loss).asscalar()
        train_acc += Cal_Acc(output,label)

    test_acc = evaluate_acc(test_data, net)
    print '%d epoch: the TRAIN loss is %f, the TRAIN accuracy is %f, the TEST accuracy is %f .' %(e, train_loss / len(train_data), train_acc / len(train_data), test_acc)
