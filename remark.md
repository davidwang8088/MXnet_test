##  x.mean()笔记
    a = np.arange(0,24).reshape((2,3,4))
    print b = a.mean(axis = (0,1),keepdims = True).shape
结果：  
      a= [[[ 0  1  2  3]  
        [ 4  5  6  7]  
        [ 8  9 10 11]]  
        [[12 13 14 15]  
        [16 17 18 19]  
        [20 21 22 23]]]  
 axis= (0,1)意味着a矩阵0、1两个维度求均值后维度值为1  
 即b.shape = (1,1,4)。  
 axis的值取几，那么均值在该维度的值就为1。  
 先有了输出矩阵的shape，就好计算了。  
