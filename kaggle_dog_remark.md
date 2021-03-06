## 2017-11-14 Remark  
### 一.跑出了第一个结果   
![kaggle_dog_1st submission](https://github.com/davidwang8088/MXnet_test/blob/master/images/kaggle_dog_1.60917.png)  
### 1.模型    
模型选用model zoo 中的resnet152_v1,DA只有rand_mirror=True.
超参数:
```
num_epochs = 150
learning_rate = 0.01
weight_decay = 5e-4
lr_period = 80
lr_decay = 0.1
```
### 2.踩到的坑  
(1)可以通过net.collect_params().save(model_dir)保存过程中产生的参数,**但是无法通过net.collect_params().load(model_dir)加载参数.**

## 2017-11-15 Todo:
* 测试基于model zoo的model,使用小epoch(e.g.epoch=10),看看提交的效果.  
    **使用了epoch=10, submission result:1.88! !便于测试!**
* 选择不同DA
  (1) model zoo resnet中的resnet152_v1, DA: 三个全开, epoch=10. submission result:2.147.  
  三个DA会发生欠拟合的情况,看到log时就感觉10个epochggo够呛能收敛.  
  现在epoch变为20,继续测试.  
* 使用model zoo densenet
* 使用shelock densenet,查看与model zoo的差距.
* 解决net.collect_params().load(model_dir)的问题

## 2017-11-18remark:  
* 先在原有代码上练习核心内容,再考虑代码合理性,优美性.  
自己重写的代码没有抓住fine tuning的本质,结果导致了一个怎么也解决不了的问题,只能改回tutorial的源码. 即使改写,**也要单元测试**,否则连在一起就是个噩梦...  
* **fine tuning模型中,应该锁定net.feature的parameter,因为kaggle_dog是imagenet的一个子集,所以fine tuning得到的参数,对kaggle_dog问题是适用的,只需训练net.classifier的参数.**  
valid loss比Train loss都要低. 看得出泛化做的是不错的.  
```
#resnet50_v2 fine tuning:
num_epochs = 20
learning_rate = 0.01
weight_decay = 5e-4
lr_period = 80
lr_decay = 0.1
#training result:
Epoch 0. Train loss: 3.512373, Valid loss 1.712977, Time 00:04:02, lr 0.01
Epoch 1. Train loss: 2.674281, Valid loss 1.457443, Time 00:04:09, lr 0.01
Epoch 2. Train loss: 2.476326, Valid loss 1.492412, Time 00:04:07, lr 0.01
Epoch 3. Train loss: 2.408694, Valid loss 1.348946, Time 00:04:09, lr 0.01
Epoch 4. Train loss: 2.341582, Valid loss 1.359826, Time 00:04:08, lr 0.01
Epoch 5. Train loss: 2.266770, Valid loss 1.252112, Time 00:04:09, lr 0.01
Epoch 6. Train loss: 2.248072, Valid loss 1.211580, Time 00:04:09, lr 0.01
Epoch 7. Train loss: 2.214287, Valid loss 1.229513, Time 00:04:09, lr 0.01
Epoch 8. Train loss: 2.181048, Valid loss 1.201421, Time 00:04:09, lr 0.01
Epoch 9. Train loss: 2.174796, Valid loss 1.192285, Time 00:04:09, lr 0.01
Epoch 10. Train loss: 2.158222, Valid loss 1.230968, Time 00:04:09, lr 0.01
Epoch 11. Train loss: 2.116029, Valid loss 1.261541, Time 00:04:10, lr 0.01
Epoch 12. Train loss: 2.134877, Valid loss 1.081200, Time 00:04:08, lr 0.01
Epoch 13. Train loss: 2.149361, Valid loss 1.134528, Time 00:04:07, lr 0.01
Epoch 14. Train loss: 2.124681, Valid loss 1.137760, Time 00:04:09, lr 0.01
Epoch 15. Train loss: 2.108666, Valid loss 1.113905, Time 00:04:09, lr 0.01
Epoch 16. Train loss: 2.109114, Valid loss 1.141784, Time 00:04:09, lr 0.01
Epoch 17. Train loss: 2.093371, Valid loss 1.109152, Time 00:04:07, lr 0.01
Epoch 18. Train loss: 2.100347, Valid loss 1.101088, Time 00:04:09, lr 0.01
Epoch 19. Train loss: 2.080084, Valid loss 1.146815, Time 00:04:09, lr 0.01
```
![resnet50v2fine tuning](https://github.com/davidwang8088/MXnet_test/blob/master/images/resnet50_v2_fine_tuning_traning_result.png)

作为对比,以下是parameter全部训练的情况:
```
Epoch 0. Train loss: 4.823708, Valid loss 4.806104, Time 00:06:02, lr 0.01
Epoch 1. Train loss: 4.792918, Valid loss 4.815374, Time 00:06:11, lr 0.01
Epoch 2. Train loss: 4.789216, Valid loss 4.801067, Time 00:06:10, lr 0.01
```
只跑了2个epoch,看的出来,train loss和valid loss值都非常大,比只训练net.classifier的程序要差很多.单独的epoch时间也长很多.  

## 2017-11-20
杨培文在[gluon论坛](https://discuss.gluon.ai/t/topic/2399/108?u=davidwang)中,对炼丹的关键点进行了总结:
* 是否 resize 正确的大小
* 是否进行了正确的预处理
* 是否锁定了卷积层权重
* 是否设置了合理的模型结构，包括全连接层、Dropout 层等
* 是否设置了合理的学习率、训练代数
### trick1  
对于第1条,正是碰到的问题,原来没有考虑过input image的size,莫名其妙的kernel就死掉了...   
inception_v3的输入图像要求(299,299).  
resnet要求输入图像(224,224)  

## 2017-11-21 
* 在gluon环境下安装opencv  
昨天先试过了在gluon环境下使用pip install方法,并不好使,可能还是没有把opencv装在gluon环境下.  
今天还是在gluon环境下,使用```conda install -c https://conda.anaconda.org/menpo opencv3``` 搞定!  
要在conda环境下安装.

## 2017-11-25 第三次提交成绩
* mark  
![第三次成绩](https://github.com/davidwang8088/MXnet_test/blob/master/images/kaggle_dog_%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B.png)  
* 思路    

** 还是基于fine tuning,改进是先将image 通过dataset = fine_tuning_net.features(images), 构造全连接网络, 通过dataset去train. 但是会快很多. 二是可以灵活的拼接多个fine tuning 训练过的dataset, 起到模型融合的作用.
* 下一步改进  
** 尝试不同模型融合  
** 改进训练方式  
** 增加数据增强
** 选择不同的loss function

## 2017-11-27 单模型记录  
model | ypw's val_loss | my val_loss
----|----|----
inceptionv3 | 0.296050225385  | 0.2827(0.3082)
resnet152_v1 | 0.399359531701 | 0.3983
resnet101_v1 | 0.410383010283 | 0.4045
densenet161 | 0.418100789189  | 0.4071
densenet201 | 0.453403010964  | 0.4704
resnet50_v2 | 0.484435886145 | 0.5667
resnet50_v1 | 0.496179759502 | 0.5212
densenet169 | 0.512498702854 | 0.4339
resnet34_v2 | 0.536734519526
vgg19_bn | 0.557294445112   | 0.7232
vgg16_bn | 0.586511127651  
resnet34_v1 | 0.591432901099
densenet121 | 0.591716498137
vgg19 | 0.619780953974
vgg16 | 0.669267293066
vgg13_bn | 0.702507363632
vgg11_bn | 0.708396691829
vgg13 | 0.756541173905
resnet18_v2 | 0.761708110571
vgg11 | 0.789955694228
resnet18_v1 | 0.832537706941
squeezenet1.1 | 1.6066500321
squeezenet1.0 | 1.62178872526
alexnet | 1.77026221156

## 2017-11-28 
* 昨天发现inceptionv3计算结果与![杨培文](https://github.com/ypwhs/DogBreed_gluon)的结果差距较大(0.7 vs 0.27).  
** 发现了inception.features输出维度是(N,768),而其他的维度都是(N,2xxxx),遂考虑将nn.GlobalAvgPool2D()更换为nn.AvgPool2D((8,8))以增加输出维度,val_loss仍为0.7左右.在论坛中看到,由于mxnet版本问题,inception.features并不是完整的训练网络,半成品.后面直接连接pool效果自然差.  
** 升级mxnet版本解决. ``` pip install -U --pre mxnet-cu80==0.12.1b20171126```
* 下面2个改进:
** 使用DA
** 使用![stanford Dog Image](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* 组合结果  
|model | my test_loss|
|----|----|
|inception + resnet152 + resnet101 + densenet161 | 0.27353|


## 2017-11-30
* 跑通了Stanford数据集程序,按照代码中的流程和超参数设置  
![kaggle_dog_with_stanford_data](https://github.com/davidwang8088/MXnet_test/blob/master/images/kaggle_dog_stanford0.2811.png)  
但是还是比最好的成绩差一个数量级,不知道怎么还能再调整的好些...

## 2017-12-3
* 使用stanford数据集,使用了inceptionv3和resnet152_v1的组合model,得到了0.00365的成绩.我居然都没有注意到.还向培神提问,如何改进算法.  
![0.00365](https://github.com/davidwang8088/MXnet_test/blob/master/images/kaggle_dog_stanford_0.00365.png)  

# 后记
* 跑到了kaggle的第四!很是惊喜的成绩了.收获很多:
* 借鉴fine tuning model, 组合model.features.  
* 首先fine tuned model的参数值得借鉴,所以用image,跑一遍得到features = model.features(images).  
* 其次,根据几个最好的单model成绩,尝试将其model.features拼接作为input,构造一个简单的dense net, 通过input训练

