# MXnet_test
## kaggle 房价预测调参log
## 2017.10.11 第（1）次参数设置
    k = 3
    epochs = 80
    verbose_epoch = epochs -10
    learning_rate = 0.03
    weight_decay = 0
    Dropout = 0.5
    提交成绩：0.12698
    remark：训练集合上的误差0.083158，说明还是过拟合了。wd=0，只依靠dropout。  

## 第（2）次参数设置
    k = 3
    epochs = 50
    verbose_epoch = epochs -10
    learning_rate = 0.02
    weight_decay = 400
    Dropout = 0.5
    **提交成绩：0.11692**  
    remark：增大wd，降低过拟合，略增大lr（与0.01比较）降低training loss。
## 第（3）次参数设置（2017.10.12）
    k = 3
    epochs = 50
    verbose_epoch = epochs -10
    learning_rate = 0.019
    weight_decay = 480  
    Dropout = 0.5
    **提交成绩：0.11474** 
    remark：继续增大wd，调低lr，大方向是在此基础上继续调低lr，升高wd。




## 一些体会
不要一次上很多层hidden layer，上的太多在调试的时候会十分的困难。重视hidden layer的unit数量，这是原来忽略的地方，原来只是盲目增加hidden layer层数。  
并没有采用feature engineering，主要是太麻烦了，其实是不会哈。但我觉得，由于采用nn，所以数据不全的特征无需人工剔除。
为什么对label取log？？
