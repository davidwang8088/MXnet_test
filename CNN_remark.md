# CNN笔记  
在MXnet环境下：  
输入输出数据格式是 batch x channel x height x width， 
权重格式是 num_filter x channel x height x width，这里input_filter和output_filter都是1。  
data: (__batch_size__, channel, height, width)  
weight: (__num_filter__, channel, kernel[0], kernel[1])  
bias: (num_filter,)  
out: (batch_size, num_filter, out_height, out_width).  
其中，output的channel取决于weight的num_filter，batch_size为data的batch_size。  
计算方法：out[n,i,:,:]=bias[i]+∑j=0 channel data[n,j,:,:]⋆weight[i,j,:,:]，    
__先在每个channel对应的data和weight卷积，后相加，得到一个out[n,i]__


    w = nd.ones(8).reshape((4,2,1,1))
    b = nd.zeros(4)
    data = nd.arange(54).reshape((3,2,3,3))
    out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
    
    
# 一个trick  
当卷积核为(1,1)时，就是data在in_channel方向上每个元素×weight后相加。  
