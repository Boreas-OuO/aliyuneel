import numpy as np
import pandas as pd
import os 
import time
class LogConfig():
    def __init__(self,
                 row_name,
                 stock_range,
                 file_name) -> None:
        self.row_name = row_name
        self.file_name = file_name
        self.file_path = './result/'+file_name+'.csv'
        self.exist_flag = os.path.exists(self.file_path)
        if self.exist_flag:
            self.df = pd.read_csv(self.file_path,index_col=0)
        else:
            self.df = pd.DataFrame(index=self.row_name,columns=[f'stock {stock_index}' for stock_index in range(stock_range)])
            
    def append_log(self,
                   col_name,
                   data):
       self.df[col_name] = data
       
    def save_log(self,file_name):
        self.file_path = './result/'+file_name+'.csv'
        if not self.exist_flag:
            self.df.to_csv(self.file_path)
        else:
            self.df.to_csv(self.file_path,mode='a',header=False)

        

    