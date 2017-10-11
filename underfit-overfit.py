# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

num_train = 100
num_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

x = nd.random.normal(shape=(num_train + num_test, 1))
X = nd.concat(x, nd.power(x, 2), nd.power(x, 3)) # power(x,2)表示x中所有元素2次方
# y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
y = true_w[0] * X[:, 0] + true_b
y += .1 * nd.random.normal(shape=y.shape)
y_train, y_test = y[:num_train], y[num_train:]

# matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

# def test(net, X, y):
#     return square_loss(net(X), y).mean().asscalar()

def train(X_train, X_test, y_train, y_test):

    #构造单层nn
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    # 一些参数
    batch_size = min(10, y_train.shape[0])
    learning_rate = 0.01
    epoch = 100
    #读取数据
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    dataset_test = gluon.data.ArrayDataset(X_test, y_test)
    data_iter_test = gluon.data.DataLoader(dataset_test, batch_size, shuffle=False)

    #训练器
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':learning_rate})
    square_loss = gluon.loss.L2Loss()

    train_loss_list = []
    test_loss_list = []
    #训练过程
    for e in range(epoch):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss_list.append(square_loss(net(X_train), y_train).mean().asscalar())####这里没有考虑到，确实直接用所有的训练数据直接算output,而不用考虑batch的问题。
        test_loss_list.append(square_loss(net(X_test), y_test).mean().asscalar())
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.legend(['train', 'test'])
    plt.show()



# # fit
# train(X[:num_train, :], X[num_train:, :], y[:num_train], y[num_train:])
# #
# # # unferfit:仅保留X一次项
# # train(X[:num_train, :1], X[num_train:, :1], y[:num_train], y[num_train:])
# # #underfit: 保留X和X平方项
# # train(X[:num_train, :2], X[num_train:, :2], y[:num_train], y[num_train:])
#
# #underfit: 保留X2次和3次项
# train(X[:num_train, 1:3], X[num_train:, 1:3], y[:num_train], y[num_train:])
#
# train(X[:num_train, 1:2], X[num_train:, 1:2], y[:num_train], y[num_train:])
#
# train(X[:num_train, 0:2], X[num_train:, 0:2], y[:num_train], y[num_train:])

# train(X[:num_train, :], X[num_train:, :], y[:num_train], y[num_train:])

#Overfit: 当样本数量<=12时，泛化误差大于训练误差，二者无法收敛至同一值。
train(X[:num_train, :], X[num_train:, :], y[:num_train], y[num_train:])