# -*- coding: utf-8 -*-

import mxnet.ndarray as nd
import numpy as np
import pandas as pd

# x = nd.arange(0,12).reshape((3,4))
# print x
#
# exp = nd.exp(x)
# print exp
# sum_exp = nd.sum(exp, axis=1, keepdims=True) #keepdims必须为True
# print sum_exp
# print exp / sum_exp


# a = np.array([[1, 2], [3, 4],[7,8]])
# b = np.array([[5, 6, 0]])
#
# c = np.concatenate((a, b.T), axis=1)
#
# # d = np.concatenate((a,b), axis=0)
#
# print c
#
# # print c

s1 = ['a', 1.0, np.nan]
print s1
print pd.get_dummies(s1)

print 'a'