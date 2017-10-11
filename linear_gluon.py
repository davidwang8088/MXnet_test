# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)


#构造数据
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset,batch_size,shuffle = True)


##构造模型
net = gluon.nn.Sequential() #Sequential 是一个容器
net.add(gluon.nn.Dense(1))#‘1’表示输出节点的个数
net.initialize()#一定要初始化,不同于param.attach_grad()方法



##构造损失函数
square_loss = gluon.loss.L2Loss()


#训练器
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})


#训练
epochs = 5
# learning_rate = 0.0001
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            print 'net grad:      ',net[0].weight.grad()
            loss = square_loss(label,output)
        loss.backward()
        # print net[0].weight.grad()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()

    print('the %d epoch: average loss is %f' %(e, total_loss / num_examples))

dense = net[0]
print true_w, dense.weight.data()
print true_b, dense.bias.data()
print dense.weight.grad()

