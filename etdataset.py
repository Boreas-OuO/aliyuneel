import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn

class DFTDataset(Dataset):
    def __init__(self, 
                 data, 
                 window_size=96,
                 O_size=96,
                 flip=True,
                 norm=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.window_size = window_size
        self.O_size = O_size
        self.flip = flip
        self.dataset_len = len(data) - window_size - O_size + 1
        self.slicesX_ = np.zeros((self.dataset_len, window_size))
        self.slicesT_ = np.zeros((self.dataset_len, O_size))

        # 构建输入和目标序列
        for i in range(self.dataset_len):
            self.slicesX_[i] = self.data[i:i + window_size]
            self.slicesT_[i] = self.data[i + window_size:i + window_size + O_size]
        if norm:
            self.minmaxnorm()
        else:
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
    def minmaxnorm(self):
        self.slicesX = (self.slicesX_ - self.slicesX_.min()) / (self.slicesX_.max() - self.slicesX_.min())
        self.slicesT = (self.slicesT_ - self.slicesT_.min()) / (self.slicesT_.max() - self.slicesT_.min())
        
    def inverse_minmaxnorm(self, data):
        return (data * (self.slicesX_.max() - self.slicesX_.min())) + self.slicesX_.min()

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
    
def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)

    etdata = pd.read_csv('./ETTm2.csv')
    ot = etdata['OT'].values


    dft_dataset = DFTDataset(ot)

    dataset_size = len(dft_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    dataloader = torch.utils.data.DataLoader(dft_dataset, 
                                             batch_size=32, 
                                             shuffle=False,
                                             pin_memory=True)

    model = MAN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss, batch_loss, vali_loss, test_loss = 0,0,0,0
        norm_train_loss, norm_vali_loss, norm_test_loss = 0,0,0
        count=0
        for inputs, targets in tqdm(dataloader, 
                                    desc=f'Epoch {epoch + 1}/{num_epochs}',
                                    ncols=100, 
                                    leave=False):
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            if count< train_size:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                invtrans_outputs = dft_dataset.inverse_minmaxnorm(outputs)
                invtrans_targets = dft_dataset.inverse_minmaxnorm(targets)
                batch_loss = criterion(invtrans_outputs, invtrans_targets)
                norm_train_loss += loss.item()
                running_loss += batch_loss.item() * inputs.size(0)
            elif count>= train_size and count< train_size+val_size:
                model.eval()
                outputs = model(inputs)
                norm_vali_loss += criterion(outputs,targets).item()
                invtrans_outputs = dft_dataset.inverse_minmaxnorm(outputs)
                invtrans_targets = dft_dataset.inverse_minmaxnorm(targets)
                loss = criterion(invtrans_outputs, invtrans_targets)
                vali_loss += loss.item() * inputs.size(0)
            elif count>= train_size+val_size:
                model.eval()
                outputs = model(inputs)
                norm_test_loss += criterion(outputs,targets).item()
                invtrans_outputs = dft_dataset.inverse_minmaxnorm(outputs)
                invtrans_targets = dft_dataset.inverse_minmaxnorm(targets)
                loss = criterion(invtrans_outputs, invtrans_targets)
                test_loss += loss.item() * inputs.size(0)
            count+=inputs.size(0)
        epoch_loss = running_loss / train_size
        vali_loss = vali_loss / val_size
        test_loss = test_loss / test_size
        norm_train_loss /= train_size
        norm_vali_loss /= val_size
        norm_test_loss /= test_size
        print(f"Epoch {epoch + 1}/{num_epochs}, Train_Loss: {epoch_loss:.4f}/{norm_train_loss:.4f}, \
              Vali_Loss:{vali_loss}/{norm_vali_loss:.4f}, Test_Loss:{test_loss}/{norm_test_loss:.4f}")
        if epoch %10 ==0:
            checkpoint = {'epoch':epoch,
                          'model':model.state_dict(),
                          'optimizer':optimizer.state_dict()}
            torch.save(checkpoint,f"./save_models/model_{epoch}.pth")
    print("Finished Training")
    
if __name__ == '__main__':
    main()