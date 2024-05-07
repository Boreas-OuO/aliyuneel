import numpy as np
import einops

class Datapre():
    def __init__(self,
                 data,
                 window_len=12,
                 flip=True) -> None:
        self.data_all = data
        self.window_len = window_len
        self.flip = flip
        

    
    def build_matrix(self,sx):
        ma = np.zeros((len(sx),len(sx)))
        for i in range(len(sx)):
            ma[:(i+1),i] = sx[:i+1][::-1]
            ma[i,:(i+1)] = sx[:i+1][::-1]
        return np.array([ma])
    
    def build_linear_matrix(self,sliceX):
        Matrix_X = np.zeros((sliceX.shape[0],1,sliceX.shape[1],sliceX.shape[1]))
        for mi in range(sliceX.shape[0]):
            Matrix_X[mi,0] = self.build_matrix(sliceX[mi])
        
    def series2slices(self,series):
        """
        :param series: 输入的完整序列
        :param window_len: 默认为12
        :return:
            slicesX: series长度*window_len的矩阵
                sliceX的第一行为从第一天开始的数据，第二行是第一行数据左移一格，即drop第一天引入第十三天，按照这个顺序往下走
            slicesT: series长度-window_len长度的向量
                从第十三天开始，一直到最后一天的数据
        """
        data_len = series.shape[0]
        # sample_len: 数据的长度-12
        sample_len = data_len-self.window_len
        # sliceX是一个矩阵，有sample_len行，window_len=12列
        slicesX = np.zeros([sample_len,self.window_len])
        slicesT = np.zeros([sample_len,1])
        for i in range(self.window_len):
            slicesX[:,i] = series[i:i+sample_len]
        slicesT[:,0] = series[self.window_len:]

        return slicesX, slicesT
    
    def slices2DFTtri(self, 
                      slicesX, 
                      without_f0=False):
        timeSteps = slicesX.shape[0]
        windowSize = slicesX.shape[1]
        max_freq = int((windowSize + 3) / 2)  # int()函数只保留整数部分
        # complex128就是64+64的复数
        # timestep是矩阵的数量，后面两个参数决定了每个矩阵的维度，生成1247个12*12的matrix
        DFTtri = np.zeros([timeSteps, windowSize, windowSize], dtype = np.complex64) #降低精度防止爆内存
        for i in range(timeSteps):
            for j in range(windowSize):
                fft = np.fft.fft(slicesX[i, -(1+j):])
                DFTtri[i, :(j+1), j] = fft[::-1]
                if self.flip:
                    DFTtri[i, j, :(j+1)] = fft[::-1] # Flip padding
        if without_f0:
            DFTtri = DFTtri[:,1:,1:]

        return DFTtri
    def detrend(self,
                slicesX,
                slicesT,
                degree=2):
        poly_degree = degree+1
        window_len = slicesX.shape[1] # (1246, 12)
        window_pattern = np.zeros([poly_degree,window_len]) # (3, 12)

        for i in range(poly_degree):
            window_pattern[i,:] = np.arange(1-window_len,1)**i
        moment_estimate_matrix = np.dot(
            window_pattern.T,
            np.linalg.inv(np.dot(window_pattern,window_pattern.T))
            )
        
        TrendDatas = np.dot(slicesX,moment_estimate_matrix)
        
        deTrendedSlicesX = np.zeros_like(slicesX)
        deTrendedSlicesX = slicesX-np.dot(TrendDatas,window_pattern)
        
        deTrendedSlicesT = np.zeros_like(slicesT)
        deTrendedSlicesT = slicesT-TrendDatas.sum(axis=1, keepdims=True)
        
        return deTrendedSlicesX, deTrendedSlicesT, TrendDatas
    
    def nd2bin(self,slicesT):
        bin_target = np.array([[int(i>0)] for i in slicesT])
        return bin_target
        
    def get_data(self,
                 the_number_of_stock,
                 bin=False,
                 linear=False):
        self.the_number_of_stock = the_number_of_stock
        index = np.zeros([600,])
        # 创建长度为600的零数组names_list
        names_list = np.zeros([600,], dtype = object)  # object是通用数据类型
        pos = 0
        for i in range(len(self.data_names)-1):  # 这里的data_names还没drop_duplicates
            # 如果第i个名字不等于第i+1个，即到了一个新的股票
            if self.data_names[i] != self.data_names[i+1]:
                # 就把names_list对应的位置替换成i股票的最后一个
                names_list[pos] = self.data_names[i]
                # 记录下来此处的位置和index
                pos += 1
                index[pos] = i+1
        # 因为他生成了600个空元素，后面发现只有504个元素，所以去除掉0 
        names_list = names_list[:pos+1]
        index[pos+1] = len(self.data_names)-1
        index = index[:pos+2]
        names_index = {names_list[i]:[int(index[i]),int(index[i+1])] for i in range(len(names_list)-1)}
        self.stock_name = names_list[the_number_of_stock]
        begin = names_index[names_list[self.the_number_of_stock]][0]
        end = names_index[names_list[self.the_number_of_stock]][1]
        data_current = self.data_all[begin:end]
        date_time_current = self.date_time[begin:end]

        data_used = np.log(data_current['close'] / data_current['close'].shift(1))
        data_used = data_used[1:]
        #TODO: COnsider a longger sampling step or other sampling strategy

        slicesX, slicesT = self.series2slices(data_used)
        originalDFTtri = self.slices2DFTtri(slicesX)
        if linear:
            self.linear_matrix = self.build_linear_matrix(slicesX)
        self.sliceX = slicesX
        self.sliceT = slicesT
        
        DFTtriPolar = np.stack([np.abs(originalDFTtri), np.angle(originalDFTtri)], axis=-1) # (1246, 12, 12, 2)
        DFTtriDescartes = np.stack([originalDFTtri.real, originalDFTtri.imag], axis=-1) # (1246, 12, 12, 2)
        deTrendedSlicesX, deTrendedSlicesT, TrendDatas = self.detrend(slicesX, slicesT)
        pytorch_DFTtriDescartes = einops.rearrange(DFTtriDescartes, 'b h w c -> b c h w')
        if bin:
            bin_sliceT = self.nd2bin(slicesT)
            return pytorch_DFTtriDescartes, bin_sliceT
        return pytorch_DFTtriDescartes, slicesT
        
        