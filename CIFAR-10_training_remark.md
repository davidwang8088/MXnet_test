## CIFAR-10调试
第一遍训练完，training acc=1，test acc = 0.91。  
有些过拟合，应对过拟合的方法：
1、调整wd(补充:在cifar10中,wd并不是一个调整后作用明显的参数,而且wd值本身也非常之小,与房价预测kaggle不同,cifar10调参主要还是网络结构的选择  
ResNet,DenseNet...,确定了网络结构后,再调整lr和lr_period,lr_decay,epoch)
2、增加image agumentation
3、减小学习率
4、提前终止。


##  第二次提交0.93010
还是res18结构，对于cifar10结构有些小，尤其实在图像增强后。  
与1st提交相比，增加了epoch，加入图像增强，泛化能力增强，epoch可能还要再增加。  
或者更换更深的模型。

## 下一步：  
### 画出loss和accuracy的曲线图  
### 竞价实例的尝试
### resnet和dense net 尝试
### 参数存储  

## 几个可以尝试的方向:  
### 1.调整resnet18的参数,e.g. node数量... 
### 2.使用resnet164  
### 3.选择densenet

--------------------------------------------------------------
## 虽然比赛结束了,但是继续debug,2017-11-06  
cifar10主要还是自己手慢了,没有进入前20.这次比赛才更有调参的感觉.  
取得成绩的模型都是基于resnet18跑出来的,没来的及上resnet164和densenet164,以及ensemble these.  
出差10天只调出了一个resnet18的结果.
----------------------------------------------------------------  
## 调试ensemble记录  
1. 原来写了句  
    ctx = mx.gpu(1)  
跑到加载test数据时,kernel就dead...问题就出在上面这句.只有一个gpu(0).  
还是用utils中  
    ctx = try_gpu()  
来搞定吧.

### 2017-11-09 记录  
昨天学习@yinglang code里面的DenseNet,网络结构打印出来是乱的,先升级了mxnet至0.12,搞不定.把kernel换为python3,搞定!  
后来发现其实在gluon下,使用的就是python3,只不过在python notebook中让我强行选择了python2.  

### Kaggle终于提交了最终成绩:0.9686!  
![image](https://github.com/davidwang8088/MXnet_test/raw/master/images/kaggle_cifar10_0.9686.png)
三种ensemble方法:sum,softmax_sum,biggest.  
最好的效果是sum方法.   
具体的,选择了6个单项成绩最好的算法,组成了perdition[6]数组,数组中每个元素就是单项提交的prediction.  
分别采用上面的三种方法进行组合.最终提交的成绩显示sum成绩最优!  

另外,组合时每个元素乘其单项提交的成绩,我特意选择所有成绩都为1,计算得到sum__的成绩,发现与sum成绩完全一致!  

