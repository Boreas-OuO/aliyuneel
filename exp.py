import pandas as pd
import numpy as np
from src.dcv3 import DFTTRI

def run(index,name,pred_len,lr = 1e-3,num_epochs=100):
    etdata = pd.read_csv(f'./dataset/{name}/{name}.csv')
    ot = etdata['OT'].values
    #_ot = np.arange(0, 10000 ,dtype=np.float32)
    #ot = np.sin(_ot)
    pred_len = pred_len

    batch_size = 512

    dfttri = DFTTRI(num_epochs=num_epochs , pred_len=pred_len, lr=lr)
    dfttri.load_data(ot, name=name, batch_size=batch_size,index=index)
    exp_index = dfttri.train()
    return exp_index

if __name__ == '__main__':
    namelist = [
            'electricity',
            'ETTm2',
            'exchange_rate',
            #'illness',
            'traffic',
            'weather']

    pred_len_list = [96, 192, 336, 720]
    index = 0
    for name in namelist:
        for pred_len in pred_len_list:
            exp_index = run(index,name,pred_len=pred_len,lr=1e-3,num_epochs=300)
            index = exp_index
