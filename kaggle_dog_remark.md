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
    **使用了epoch=10, submission result:1.88! 这个性价比已经很高了!便于测试!**
* 选择不同DA
* 使用model zoo densenet
* 使用shelock densenet,查看与model zoo的差距.
* 解决net.collect_params().load(model_dir)的问题
