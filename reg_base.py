# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import random
import matplotlib.pyplot as plt

num_train = 20
num_test = 100
num_inputs = 200

true_w = nd.ones((num_inputs,1)) * 0.01
true_b = 0.05

X = nd.random.normal(shape = (num_train+num_test, num_inputs))
y = nd.dot(X,true_w) + true_b
y += 0.01 * nd.random.normal(shape = y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]

batch_size = 1
def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples,batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield X.take(j), y.take(j)


def get_params():
    w = nd.random.normal(shape = (num_inputs,1))*0.1
    b = nd.zeros((1,))
    for param in (w,b):
        param.attach_grad()
    return (w, b)


# L2
def L2_norm(w,b):
    return (w**2).sum() + b**2


def net(x, lambd, w, b):
    return nd.dot(x,w) + b

def square_loss(output, label):
    return (output-label.reshape(output.shape))**2

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def train(lambd):
    epoch = 10
    learning_rate = 0.002 ###！！！！调来调去，结果loss不收敛，绝对原因就是learning rate 设置的不合理，过大，宁愿一开始小点收敛慢些！！！！！！！！
    train_loss_list = []
    test_loss_list = []

    w,b = get_params()
    for e in range(epoch):
        for data, label in data_iter(num_train):
            with autograd.record():
                yhat = net(data,lambd,w,b)
                loss = square_loss(yhat,label) + lambd * L2_norm(w,b)
            loss.backward()
            SGD((w,b),learning_rate)

        loss_train = square_loss(net(X_train,lambd,w,b),y_train)
        # print loss_train.mean().asscalar()
        train_loss_list.append(loss_train.mean().asscalar())
        loss_test = square_loss(net(X_test,lambd,w,b),y_test)
        test_loss_list.append(loss_test.mean().asscalar())

    # print train_loss_list
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.legend(['train', 'test'])
    plt.show()
    print 'w[:10,:]:  ', w[:10,:]


train(0)
train(10)