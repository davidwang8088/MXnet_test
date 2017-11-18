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
* 自己重写的代码没有抓住fine tuning的本质,结果导致了一个怎么也解决不了的问题,只能改回tutorial的源码. 即使改写,**也要单元测试**,否则连在一起就是个噩梦...  
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

