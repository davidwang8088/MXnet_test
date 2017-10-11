# -*- coding: utf-8 -*-

from mxnet import nd
import mxnet as mx
# w = nd.arange(4).reshape((1,1,2,2))
# b = nd.array([1])
# data = nd.arange(9).reshape((1,1,3,3))
# out = nd.Convolution(data, w, b, kernel = w.shape[2:], num_filter = w.shape[1])
#
# print data, w, b, out
#
# output = nd.Convolution(data,w,b,kernel = w.shape[2:],num_filter = w.shape[1],pad=(1,1),stride = (2,2))

# print output

# w = nd.arange(16).reshape((2,2,2,2))# 第一个维度是每个通道filter个数，第二个维度是通道数，后面两个是每个filter的维度
# data = nd.arange(18).reshape((100,2,3,3))
# b = nd.array([1,2])
#
#
# out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
# print w.shape[2:]
# print w.shape[0]
# print w
# print out
#
#
# w = nd.arange(8).reshape((1,2,2,2)) # w = (a,b,c,d),其中a代表输出的通道数，b代表当前输入的通道数，[c,d]代表kernel大小
# data = nd.arange(72).reshape((4,2,3,3)) #data=(e,f,g,h), 其中e代表样本数量，f表示样本通道数，[g,h]代表data一个通道的数据大小
# b = nd.array([1])
# out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
# # print data, w,
# print out.shape

# data = nd.arange(18).reshape((1,2,3,3))
# max_pool = nd.Pooling(data = data, pool_type = "max", kernel = (2,2))
# avg_pool = nd.Pooling(data = data, pool_type = "avg", kernel = (2,2))
# print 'data', data
# print 'max_pool', max_pool
# print 'avg_pool',avg_pool


try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
ctx

import sys
sys.path.append('..')
from utils import load_data_fashion_mnist

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)

weight_scale = .01
num_outputs = 10
num_fc = 128



W1 = nd.random_normal(shape = (20,1,5,5), scale = weight_scale, ctx = ctx)
b1 = nd.zeros(W1.shape[0], ctx=ctx)

W2 = nd.random_normal(shape = (50,20,3,3), scale = weight_scale, ctx = ctx)
b2 = nd.zeros(W2.shape[0],ctx=ctx)

W3 = nd.random_normal(shape = (1250,128), scale = weight_scale, ctx = ctx)
b3 = nd.zeros(W3.shape[1],ctx = ctx)

W4 = nd.random_normal(shape= (W3.shape[1],10), scale = weight_scale, ctx = ctx)
b4 = nd.zeros(10, ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]
for param in params:
    param.attach_grad()


def net(X, Verbose = False):
    X = X.as_in_context(W1.context) #将X的存储位置与W1一致
    #第一层卷积
    h1_conv = nd.Convolution(data = X, weight = W1, bias = b1, kernel = W1.shape[2:], num_filter = W1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data = h1_activation, pool_type = "max", kernel = (2,2), stride = (2,2))
    #第二层卷积
    h2_conv = nd.Convolution(data = h1, weight = W2, bias = b2, kernel = W2.shape[2:], num_filter = W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data = h2_activation, pool_type = "max", kernel = (2,2), stride = (2,2))
    h2 = h2.flatten()
    #第三层全链接
    h3 = nd.relu(nd.dot(h2, W3) + b3)
    #第四层全链接
    h4 = nd.dot(h3,W4)+b4
    if Verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4.shape)
        print('output:', h4)
    return h4

# for data, _ in train_data:
#     net(data, Verbose = True)
#     break

from mxnet import autograd as ag
from utils import SGD, accuracy, evaluate_accuracy
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))