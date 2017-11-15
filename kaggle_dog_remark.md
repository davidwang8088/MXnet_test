## 2017-11-14 Remark  
### 跑出了第一个结果   
![kaggle_dog_1st submission](https://github.com/davidwang8088/MXnet_test/blob/master/images/kaggle_dog_1.60917.png)  
### 模型    
模型选用model zoo 中的resnet152_v1,DA只有rand_mirror=True.
超参数:
```
num_epochs = 150
learning_rate = 0.01
weight_decay = 5e-4
lr_period = 80
lr_decay = 0.1
```
### 踩到的坑  

