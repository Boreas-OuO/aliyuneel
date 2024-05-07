import pandas as pd

class DetailLog():
    def __init__(self,
                 test_size,
                 O_size,
                 file_name):
        self.test_size = test_size
        self.O_size = O_size
        self.col_names = [f'{i//2}_predict' if i%2==0 else f'{(i-1)//2}_real' for i in range(test_size*2)]
        self.index_name = [i for i in range(O_size)]
        self.alllog = pd.DataFrame(index=self.index_name,columns=self.col_names)
        self.file_name = file_name
        self.file_path = './result/'+file_name+'_detail.csv'
        pass
    
    def append_log(self,
                   col_index,
                   predict,
                   real):
        self.alllog[f'{col_index}_predict'] = predict
        self.alllog[f'{col_index}_real'] = real

    def save_log(self):
        self.alllog.to_csv(self.file_path)
        pass