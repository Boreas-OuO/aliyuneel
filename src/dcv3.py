import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class DFTDataset(Dataset):
    def __init__(self, 
                 data, 
                 window_size=96,
                 O_size=720,
                 flip=True,
                 norm=True,
                 mean=0,
                 std=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not mean:
            self.mean = data.mean()
            self.std = data.std()
        else:
            self.mean = mean
            self.std = std
        if norm:
            self.data = self.transform(data)
        else:
            self.data = Data
        self.window_size = window_size

        self.O_size = O_size
        self.flip = flip
        self.dataset_len = len(self.data) - window_size - O_size + 1
        self.slicesX_ = np.zeros((self.dataset_len, window_size))
        self.slicesT_ = np.zeros((self.dataset_len, O_size))


        # 构建输入和目标序列
        for i in range(self.dataset_len):
            self.slicesX_[i] = self.data[i:i + window_size]
            self.slicesT_[i] = self.data[i + window_size:i + window_size + O_size]
        
        self.slicesX = self.slicesX_
        self.slicesT = self.slicesT_
        self.input_DFTtri = self.slices2DFTtri(self.slicesX)
        self.slicesT = self.slicesT
        #self.input_DFTtri = ei.rearrange(self.slices2DFTtri(self.slicesX),'b h w c -> b c h w')
        

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        input_DFTtri = self.input_DFTtri[idx]
        target_seq = self.slicesT[idx]
        
        return input_DFTtri, target_seq
    def slices2DFTtri(self, 
                      slicesX, 
                      without_f0=False):
        timeSteps = slicesX.shape[0]
        windowSize = slicesX.shape[1]
        max_freq = int((windowSize + 3) / 2)  # int()函数只保留整数部分
        # complex128就是64+64的复数
        # timestep是矩阵的数量，后面两个参数决定了每个矩阵的维度，生成1247个12*12的matrix
        DFTtri = np.zeros([timeSteps, windowSize, windowSize], dtype = np.complex64) #降低精度防止爆内存
        for i in tqdm(range(timeSteps),desc="Loading Data",ncols=100):
            for j in range(windowSize):
                fft = np.fft.fft(slicesX[i, -(1+j):])
                DFTtri[i, :(j+1), j] = fft[::-1]
                if self.flip:
                    DFTtri[i, j, :(j+1)] = fft[::-1] # Flip padding
        if without_f0:
            DFTtri = DFTtri[:,1:,1:]
        DFTtriDescartes = np.stack([DFTtri.real, DFTtri.imag], axis=-3)
        return DFTtriDescartes
    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



class MAN(nn.Module):
    def __init__(self, num_classes=96):
        super(MAN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Best:
    def __init__(self, path, name,pred_len,index=0):
        self.name = name
        self.pred_len = pred_len
        self.load_template()
        self.index = 0
        self.best_path = path + f'exp_{self.index}/'
        if index==0:
            while os.path.exists(self.best_path):
                self.index+=1 
                self.best_path = path + f'exp_{self.index}/'
                # print(self.index)
            os.mkdir(self.best_path)
            os.mkdir(self.best_path+"best_models/")
            self.log = self.Template
            self.path = self.best_path + 'log.json'
        else:
            self.index = index
            self.best_path = path + f'exp_{self.index}/'
            self.path = self.best_path + 'log.json'
            self.log = self.load_log()
        print(f"Experient Index: {self.index}")
        
        self.vali_best = 100
    
    def get_exp_index(self):
        return self.index

    def load_log(self):
        with open(self.path, 'r') as file:
            data = json.load(file)
        return data

    def load_template(self):
        Template_path = './Template/exp_log.json'
        with open (Template_path, 'r') as f:
            self.Template = json.load(f)
    
    """
    def save_test_log(self,  test_loss, mse, mae, epoch):
        if test_loss < self.test_best:
            self.test_best = test_loss
            self.test_log[f"{self.name}"][f"{self.pred_len}"]["MSE"] = mse
            self.test_log[f"{self.name}"][f"{self.pred_len}"]["MAE"] = mae
            with open(self.test_path, 'w') as f:
                json.dump(self.test_log, f, indent=4)
    """    
            
    def save_log(self, vali_loss,v_mse,v_mae,t_mse,t_mae,epoch,model):
        if vali_loss < self.vali_best:
            self.vali_best = vali_loss
            self.log[f"{self.name}"][f"{self.pred_len}"]["EPOCH"] = epoch
            self.log[f"{self.name}"][f"{self.pred_len}"]["MSE_VALI"] = v_mse
            self.log[f"{self.name}"][f"{self.pred_len}"]["MAE_VALI"] = v_mae
            self.log[f"{self.name}"][f"{self.pred_len}"]["MSE_TEST"] = t_mse
            self.log[f"{self.name}"][f"{self.pred_len}"]["MAE_TEST"] = t_mae
            with open(self.path, 'w') as f:
                json.dump(self.log, f, indent=4)
            checkpoint = {
                "epoch":epoch,
                "model_state_dict": model.state_dict()
            }
            if self.vali_best!=100:
                torch.save(checkpoint,self.best_path+f"best_models/{self.name}_{self.pred_len}.pth")
            return True
        return False




class ComputeLoss:
    def __init__(self):
        self.mse_loss, self.mae_loss, self.inv_mse_loss, self.inv_mae_loss = 0, 0, 0, 0
    def reset_loss(self):
        self.mse_loss, self.mae_loss, self.inv_mse_loss, self.inv_mae_loss = 0, 0, 0, 0
    
    def update_loss(self,ori_loss,inv_loss,batch_size):
        self.mse_loss += ori_loss[0]
        self.mae_loss += ori_loss[1]
        self.inv_mse_loss += inv_loss[0]
        self.inv_mae_loss += inv_loss[1]
    
    def epoch_loss(self,setlen):
        self.mse_loss /= setlen
        self.mae_loss /= setlen
        self.inv_mse_loss /= setlen
        self.inv_mae_loss /= setlen
        return self.mse_loss,self.mae_loss,self.inv_mse_loss,self.inv_mae_loss
        
                       

class DFTTRI:
    def __init__(self, pred_len, num_epochs=50, lr=1e-3,log_path='./save_log/'): 
        self.pred_len = pred_len  
        self.log_path = log_path                                                                          
        self.model = MAN(num_classes=pred_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)     
        self.num_epochs = num_epochs
        self.writer = SummaryWriter('logs')
        self.epoch = 0
        self.loss = 0
        self.lazy_threshold = 50
        

    def load_data(self, ot, name='edt', batch_size=256, index=0):
        self.dataset_name = name
        self.name = name + f'-{self.pred_len}'
        dataset_size = ot.shape[0]
        self.train_size = int(0.7 * dataset_size)
        self.val_size = int(0.1 * dataset_size)
        self.test_size = dataset_size - self.train_size - self.val_size
        self.load_dataloader(ot,batch_size)
        self.best = Best(self.log_path,name,pred_len=self.pred_len,index=index)


    def mse(self, outputs, targets):
        return ((outputs - targets) ** 2).mean().item()

    def mae(self, outputs, targets):
        return (outputs - targets).abs().mean().item()
    
    def metric(self, outputs, targets):
        return self.mse(outputs, targets), self.mae(outputs, targets)

    def _plot(self, outputs, targets, mse_loss,mae_loss,inv_mse_loss,inv_mae_loss, mode):
        if mode == 'train':
            data_len = len(self.trainloader)
        elif mode == 'vali':
            data_len = len(self.valiloader)
        else:
            data_len = len(self.testloader)
        self.writer.add_scalar(f'{self.name}/{mode}/Loss/mse', mse_loss, self.epoch)
        self.writer.add_scalar(f'{self.name}/{mode}/Loss/mae', mae_loss, self.epoch)
        self.writer.add_scalar(f'{self.name}/{mode}/Loss/mse_inv', inv_mse_loss, self.epoch)
        self.writer.add_scalar(f'{self.name}/{mode}/Loss/mae_inv', inv_mae_loss, self.epoch)

        final_output = outputs[-1]
        final_target = targets[-1]
        # 绘制final_output和final_target，并保存到writer的logs中
        fig, ax = plt.subplots()
        ax.plot(final_output.detach().cpu().numpy(), label='Final Output')
        ax.plot(final_target.detach().cpu().numpy(), label='Final Target')
        ax.legend()
        self.writer.add_figure(f'{self.name}/{mode}/Final_Output_vs_Target', fig, global_step=self.epoch)
    
    def save_dataloader(self,batch_size):
        save_path = f"./save_dataloader/{self.name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        torch.save({
        "trainset": self.trainset,
        "valiset": self.valiset,
        "testset": self.testset,
        "trainloader": self.trainloader,
        "valiloader": self.valiloader,
        "testloader": self.testloader
        }, f"{save_path}/{batch_size}.pth", pickle_protocol=4)
    
    def load_dataloader(self,ot,batch_size):
        save_path = f"./save_dataloader/{self.name}/{batch_size}.pth"
        self.batch_size = batch_size
        if os.path.exists(save_path):
            print(f"Loading Exist Data: {save_path}")
            saved_dataloader = torch.load(save_path)
            self.trainset = saved_dataloader['trainset']
            self.valiset =saved_dataloader['valiset']
            self.testset = saved_dataloader['testset']
            self.trainloader = saved_dataloader["trainloader"]
            self.valiloader = saved_dataloader["valiloader"]
            self.testloader = saved_dataloader["testloader"]
        else:
            self.trainset = DFTDataset(ot[:self.train_size],O_size=self.pred_len)
            mean, std = self.trainset.mean, self.trainset.std
            self.valiset = DFTDataset(ot[self.train_size:self.train_size+self.val_size],O_size=self.pred_len,mean=mean,std=std)
            self.testset = DFTDataset(ot[self.train_size+self.val_size:],O_size=self.pred_len,mean=mean,std=std)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, 
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    pin_memory=True)
            self.valiloader = torch.utils.data.DataLoader(self.valiset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False,
                                                    pin_memory=True)
            self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False,
                                                    pin_memory=True)   
            self.save_dataloader(batch_size=batch_size)                                       

    def train(self):
        self.model.train()
        optimizer = self.optimizer
        computel = ComputeLoss()
        with tqdm(total=self.num_epochs, desc=self.dataset_name, ncols=100, leave=False) as pbar:
            for self.epoch in range(self.num_epochs):
                computel.reset_loss()
                for inputs, targets in self.trainloader:
                    inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    invtrans_outputs = self.trainset.inverse_transform(outputs)
                    invtrans_targets = self.trainset.inverse_transform(targets)
                    ori_loss = self.metric(outputs, targets)
                    inv_loss = self.metric(invtrans_outputs, invtrans_targets)
                    computel.update_loss(ori_loss,inv_loss,self.batch_size)
                mse_loss,mae_loss,inv_mse_loss,inv_mae_loss = computel.epoch_loss(len(self.trainloader))
                self._plot(outputs, targets, mse_loss,mae_loss,inv_mse_loss,inv_mae_loss, 'train')
                v_l,v_mse,v_mae = self.vali()
                t_l,t_mse,t_mae = self.test()
                if self.best.save_log(v_l,v_mse,v_mae,t_mse,t_mae,self.epoch,self.model):
                    self.last_update = self.epoch
                    pbar.set_postfix(Last_Updated = self.epoch)
                pbar.set_description(f'{self.dataset_name} (Loss: {loss:.4f})')
                pbar.update(1)
                if self.epoch > self.last_update+self.lazy_threshold:
                    break
        return self.best.get_exp_index()
                    
        
    
    def vali(self):
        self.model.eval()
        computel = ComputeLoss()
        with torch.no_grad():
            for inputs, targets in self.valiloader:
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                invtrans_outputs = self.valiset.inverse_transform(outputs)
                invtrans_targets = self.valiset.inverse_transform(targets)
                ori_loss = self.metric(outputs, targets)
                inv_loss = self.metric(invtrans_outputs, invtrans_targets)
                computel.update_loss(ori_loss,inv_loss,self.batch_size)
        mse_loss,mae_loss,inv_mse_loss,inv_mae_loss = computel.epoch_loss(len(self.valiloader))          
        self._plot(outputs, targets, mse_loss,mae_loss,inv_mse_loss,inv_mae_loss, 'vali')
        return mse_loss,mse_loss,mae_loss

    def test(self):
        self.model.eval()
        computel = ComputeLoss()
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                invtrans_outputs = self.testset.inverse_transform(outputs)
                invtrans_targets = self.testset.inverse_transform(targets)
                ori_loss = self.metric(outputs, targets)
                inv_loss = self.metric(invtrans_outputs, invtrans_targets)
                computel.update_loss(ori_loss,inv_loss,self.batch_size)
        mse_loss,mae_loss,inv_mse_loss,inv_mae_loss = computel.epoch_loss(len(self.testloader))    
        self._plot(outputs, targets, mse_loss,mae_loss,inv_mse_loss,inv_mae_loss, 'test')
        return mse_loss,mse_loss,mae_loss

    def run(self):
        pass
        
    
