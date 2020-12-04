from numba import cuda, jit, float64

@cuda.jit
def mat_mul_naive_kernal(sdata, fcore, rdata):
    '''matrix multiplication on gpu, naive method using global device memory
    '''
    i, j = cuda.grid(2)
    if i+1 < rdata.shape[0] and j+1 < rdata.shape[1]:
        summation = 0
        for k in range(9):
            summation += sdata[i+k//3, j+k%3] * fcore[k]
        rdata[i+1, j+1] = summation
def host_naive(sdata, fcore, rdata):
    #主机内存数据拷贝至显存，加快运行速度
    d_A = cuda.to_device(sdata) 
    d_B = cuda.to_device(fcore)
    d_C = cuda.device_array(rdata.shape, np.float64)
    #确定块的大小
    threadsperblock = (TPB, TPB)
    #根据图像大小确定每个grid中有几个block
    blockspergrid_x = math.ceil(sdata.shape[0]/threadsperblock[0])
    blockspergrid_y = math.ceil(sdata.shape[1]/threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mat_mul_naive_kernal[blockspergrid, threadsperblock](d_A, d_B, d_C)
    
    return d_C.copy_to_host()

def Filter(data, n, core):
    y=data.shape[0]
    x=data.shape[1]
    source_data=np.zeros([y+2,x+2],dtype=np.float64)
    GFiltData=np.zeros([y+2,x+2],dtype=np.float64)
    LFiltData=np.zeros([y+2,x+2],dtype=np.float64)
    lineimage=np.zeros_like(data)
    showimage=np.zeros([y,x,3])
    
    #创建数据图像用于滤波
    source_data[0,1:x+1]=data[0,:]
    source_data[1:y+1,0]=data[:,0]
    source_data[y+1,1:x+1]=data[y-1,:]
    source_data[1:y+1,x+1]=data[:,0]
    source_data[0,0]=data[0,0]
    source_data[0,x+1]=data[0,x-1]
    source_data[y+1,0]=data[y-1,0]
    source_data[y+1,x+1]=data[y-1,x-1]
    source_data[1:y+1,1:x+1]=data[:,:]

    #创建滤波核
    Gfiltercore=np.array([0.0625,0.125,0.0625, 0.125,0.25,0.125, 0.0625,0.125,0.0625],dtype=np.float64)
    Efiltercore=np.array([-1,-1,-1,-1,8,-1,-1,-1,-1])
    start=time.time()
    #进行滤波提取边缘
    LFiltData=host_naive(source_data, Efiltercore, LFiltData)
    GFiltData=host_naive(LFiltData, Gfiltercore, GFiltData)