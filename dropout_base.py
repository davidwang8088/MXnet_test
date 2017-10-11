# -*- coding: utf-8 -*-


from mxnet import nd

def dropout(X, dropout_prob):
    keep_prob = 1 - dropout_prob
    assert 0<= keep_prob <=1
    if keep_prob == 0:
        return X.zeros_like()

    mask = nd.random.uniform(0, 1.0, X.shape, ctx=X.context) < keep_prob
    print mask
    scale = 1 / keep_prob  #
    return X * mask * scale


A = nd.arange(20).reshape((4,5))
print A
print dropout(A,0.0)
print dropout(A,0.5)
print dropout(A,1.0)