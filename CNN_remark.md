# CNN笔记  
在MXnet环境下：  
输入输出数据格式是 batch x in_channel x height x width， 
权重格式是 num_filter x in_channels x height x width，这里input_filter和output_filter都是1。  
data: (__batch_size__, channel, height, width)  
weight: (__num_filter__, channel, kernel[0], kernel[1])  
bias: (num_filter,)  
out: (batch_size, num_filter, out_height, out_width).  
其中，num_filter就是，输出的channel。即output的channel取决于weight的num_filter。  
out[n,i,:,:]=bias[i]+$\sum_{j=0}^channel$   ∑j=0channeldata[n,j,:,:]⋆weight[i,j,:,:]


$\sum_{k=1}^n$
$\sum_{k-1}^n$

    w = nd.ones(8).reshape((4,2,1,1))
    b = nd.zeros(4)
    data = nd.arange(54).reshape((3,2,3,3))
    out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
