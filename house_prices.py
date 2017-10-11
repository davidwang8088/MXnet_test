# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


# s1 = pd.Series(['a', 'b'])
# s2 = pd.Series(['c', 'd'])
# print s1
# print s2
# print pd.concat(([s1, s2]))
# print pd.concat([s1, s2])

#控制调试or测试
DEBUG = True

#测试输出submission是否出现负数
if DEBUG:
    debug_test_data = pd.read_csv("submission.csv")
    debug_test = debug_test_data.loc[:,'SalePrice']
    i = 0
    for price in debug_test:
        i += 1
        if price<=0:
            print price, i



train = pd.read_csv("data/kaggle_house_pred_train.csv")
test = pd.read_csv("data/kaggle_house_pred_test.csv")
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], #train.loc[:, 'a':'b'] 表示列标签从a列至b列的所有数据。
                      test.loc[:, 'MSSubClass':'SaleCondition'])) #pd.concat(x,y) 将两组数据按行按组合


numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index # all_X.dtypes表示其列数据类型，整条语句的意思就是选择所有列是数值类型的索引

all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean()) / x.std()) # 对数值类型的列数据，按列逐一标准化处理
#将分类变量转换为0或者1的值。具体地，比如一个列X有三个属性,外加nan：（a,b,c,nan），
#将变为4列，X_a,X-b,X-c,X_nan，样本有哪个属性就填充为1。
all_X = pd.get_dummies(all_X, dummy_na=True)
# print all_X.head()
#对数值序列，用列的均值填充nan
all_X = all_X.fillna(all_X.mean())
# print all_X.head()

num_train = train.shape[0]
x_train = all_X[:num_train].as_matrix()
x_test = all_X[num_train:].as_matrix()

# y_train = train.SalePrice
# y_train = (y_train - y_train.mean()) / y_train.std()

y_train = train.SalePrice.as_matrix()

#转换为NDarray数据
X_train = nd.array(x_train)
X_test = nd.array(x_test)
Y_train = nd.array(y_train)
Y_train.reshape((num_train,1))

square_loss = gluon.loss.L2Loss()
def get_srme_log(net, X_train, Y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(nd.log(clipped_preds), nd.log(Y_train))).asscalar() / num_train)

def direct_srme_log(output, label):
    clipped_preds = nd.clip(output, 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(nd.log(clipped_preds), nd.log(label))).asscalar() / output.shape[0])

def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(8192,activation = 'relu'))
        net.add(gluon.nn.Dropout(0.5))
        # net.add(gluon.nn.Dense(512,activation='relu'))
        # net.add(gluon.nn.Dense(256,activation='relu'))
        # net.add(gluon.nn.Dense(128, activation='relu'))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net


import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def train(net, X_train,Y_train,X_test,Y_test, epochs,
          verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
        test_loss = []
    batch_size = 100

    dataset_train = gluon.data.ArrayDataset(X_train,Y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train,batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'RMSProp', {'learning_rate': learning_rate, 'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)

    for e in range(epochs):

        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output,label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = get_srme_log(net, X_train, Y_train)
        if e > verbose_epoch:
            print 'epoach %d, current loss: %f' %(e, cur_train_loss)
        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_srme_log(net, X_test, Y_test)
            test_loss.append(cur_test_loss)

    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train', 'test'])
    plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss

def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0

    for test_i in range(k):
        x_val_test = X_train[test_i*fold_size: (test_i+1)*fold_size,:]
        y_val_test = y_train[test_i*fold_size: (test_i+1)*fold_size]

        val_train_defined = False
        for j in range(k):
            if j != test_i:
                x_cur_fold = X_train[j*fold_size:(j+1)*fold_size,:]
                y_cur_fold = y_train[j*fold_size:(j+1)*fold_size]
                if not val_train_defined:
                    x_val_train = x_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    x_val_train = nd.concat(x_val_train, x_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)

        net = get_net()
        train_loss, test_loss = train(
            net,x_val_train,y_val_train,x_val_test,y_val_test,
            epochs,verbose_epoch,learning_rate,weight_decay)
        print 'after train'
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss

    return train_loss_sum / k, test_loss_sum / k

################################################################################

k = 3
epochs = 80
verbose_epoch = epochs -10
learning_rate = 0.03
weight_decay = 2000





#预测函数
def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay):
    net = get_net()
    train(net, X_train, y_train, None, None, epochs, verbose_epoch,
          learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if DEBUG:
    train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train,
                                               Y_train, learning_rate, weight_decay)
    print "%d-fold validation: Avg train loss: %f, Avg test loss: %f" %(k, train_loss, test_loss)
else:
    learn(epochs, verbose_epoch, X_train, Y_train, test, learning_rate ,weight_decay)