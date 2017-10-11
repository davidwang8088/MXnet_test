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
dataset_train = gluon.data.ArrayDataset(X_train,y_train)
data_iter = gluon.data.DataLoader(dataset_train,batch_size,shuffle=True)


def train(weight_decay):

    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    square_loss = gluon.loss.L2Loss()

    trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':0.001,'wd': weight_decay})

    train_loss = []
    test_loss = []
    epochs = 50
    for e in range(epochs):
        for data, label in data_iter:
            with autograd.record():
                output = net(data)
                # print 'output', output
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(square_loss(net(X_train),y_train).mean().asscalar())
        test_loss.append(square_loss(net(X_test), y_test).mean().asscalar())

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()

    print ('learned w[:10]:', net[0].weight.data(),
     'learned b:', net[0].bias.data())


train(0)
train(5)




