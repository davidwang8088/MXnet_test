# -*- coding: utf-8 -*-

import mxnet.ndarray as nd
import mxnet.autograd as ag
#


num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

www = nd.array(true_w).reshape((2,1))
X = nd.random_normal(shape=(num_examples, num_inputs))
Y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
Y += 0.01 * nd.random_normal(shape = Y.shape)

print X[0],Y[0]


import random
batch_size = 10
def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i+batch_size, num_examples)])
        yield nd.take(X, j), nd.take(Y, j)


w = nd.random_normal(shape=(num_inputs,1))
b = nd.zeros((1,))
params = [w,b]
print params # w,b不一定是一样的行和列，装在一起就行

for param in params:
    # print param
    param.attach_grad()

def net(X):
    return nd.dot(X,w) + b

def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 #根据y_hat的shape调整y，防止自动广播

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


epochs = 10
learning_rate = .0003

for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        # print 'before:  ',params[0].grad
        with ag.record():
            yhat = net(data)
            loss = square_loss(yhat,label)
            # print 'loss____', loss
        # print 'after:  ',params[0].grad
        loss.backward() #如果loss不是一个标量，那么loss.backward()等价于nd.sum(loss).backward().
        SGD(params,learning_rate)

        # print nd.sum(loss)
        total_loss += nd.sum(loss).asscalar()
    print ('the %d epoch: average loss is %f' %(e, total_loss / num_examples))

print true_w, w