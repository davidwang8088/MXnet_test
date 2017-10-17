#  矩阵中axis的使用  
尤其是对于高维矩阵中X.shape = (2,3,4):  
首先确定结果矩阵的shape，axis等于几，哪个维度变为1.  
而后，再考虑计算mean，max。。。whatever...  
note:X.argmax(axis=0)输出的矩阵是坐标。
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
 
 ## X.argmax()笔记
    a = np.arange(0,24).reshape((2,3,4))  
    c = a.argmax(axis=0)  
    两个(3,4)的矩阵沿着第0维进行比较，保留第0维中大的坐标。  
    Note：a.argmax的输出要比a低一个维度。  

