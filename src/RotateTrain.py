import torch.nn as nn
import torch
import random
import numpy as np
from tqdm import tqdm


class RotateTrain():
    def __init__(self, 
                 model, 
                 data_X, 
                 data_y,
                 Rotate_size,
                 num_epochs,
                 seed,
                 lr,
                 threshold=1e-5):
        self.data_X = data_X
        self.data_y = data_y
        self.num_epochs = num_epochs
        self.seed = seed
        self.lr = lr
        self.thresold = threshold
        self.Rotate_size = Rotate_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.total_size = len(self.data_y) - self.Rotate_size - 1
        self.train_y = self.data_y[self.Rotate_size:-1]
        self.test_y = self.data_y[self.Rotate_size+1:]
        self.test_predict = np.zeros((self.total_size,))
        self.train_predict = np.zeros((self.total_size,))
        self.losslog = torch.zeros(self.num_epochs, 2)
        
    def setup_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
    def nd2bin(self,target):
        bin_target = np.array([[int(i>0)] for i in target])
        return bin_target
    
    def evaluation(self):
        pert = 1e-10
        self.bin_train_y = self.nd2bin(self.train_y)
        self.bin_test_y = self.nd2bin(self.test_y)
        self.bin_train_predict = self.nd2bin(self.train_predict)
        self.bin_test_predict = self.nd2bin(self.test_predict)
        self.true_positive = np.sum(self.bin_test_y * self.bin_test_predict)
        self.true_negative = np.sum((1-self.bin_test_y) * (1-self.bin_test_predict))
        self.false_positive = np.sum(self.bin_test_y * (1-self.bin_test_predict))
        self.false_negative = np.sum((1-self.bin_test_y) * self.bin_test_predict) 
        self.accuracy = (self.true_positive + self.true_negative) / len(self.test_y)
        self.precision = (self.true_positive+pert) / (self.true_positive + self.false_positive+pert)
        self.recall = (self.true_positive+pert) / (self.true_positive + self.false_negative+pert)
        self.f1score = (2 *self.precision * self.recall+pert) / (self.precision + self.recall+pert)
        self.rmse = np.sqrt(np.mean((self.test_y - self.test_predict)**2))
        self.mae = np.mean(np.abs(self.test_y - self.test_predict))
        self.r2 = 1 - (np.sum((self.test_y - self.test_predict)**2)+pert) / (np.sum((self.test_y - np.mean(self.test_y))**2)+pert)
        self.log = np.array([self.seed,self.mae,self.rmse,self.r2,self.accuracy,self.f1score])
    
    
    def train(self):
        self.setup_seed()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss().to(self.device)
        cut = -1
        with tqdm(total=self.total_size) as pbar:
            pbar.set_description("Training")
            pbar.update(0)
            
            for rotate_index in tqdm(range(self.total_size)):
                rotate_data_x = torch.tensor(self.data_X[rotate_index:rotate_index+self.Rotate_size],dtype=torch.float32).to(self.device)
                rotate_data_y = torch.tensor(self.data_y[rotate_index:rotate_index+self.Rotate_size],dtype=torch.float32).to(self.device)
                rotate_test_x = torch.tensor(self.data_X[rotate_index+self.Rotate_size:rotate_index+self.Rotate_size+1],dtype=torch.float32).to(self.device)
                rotate_test_y = torch.tensor(self.data_y[rotate_index+self.Rotate_size:rotate_index+self.Rotate_size+1],dtype=torch.float32).to(self.device)
                self.losslog = torch.zeros(self.num_epochs, 2)
                for epoch in range(self.num_epochs):
                    self.model.train()
                    optimizer.zero_grad()
                    output = self.model(rotate_data_x)
                    loss = criterion(output, rotate_data_y)
                    loss.backward()
                    optimizer.step()
                    ## Evaluation
                    self.losslog[epoch][0] = loss.item()
                    self.model.eval()
                    test_output = self.model(rotate_test_x)
                    loss_test = criterion(test_output, rotate_test_y)
                    self.losslog[epoch][1] = loss_test.item()
                    if loss.item() < self.thresold and epoch:
                        # self.thresold = loss.item()
                        cut = epoch
                        self.losslog = self.losslog[:cut]
                        break
                    pbar.set_postfix(test_loss=loss_test.item(),
                                        train_loss=loss.item(),
                                        cut_epoch=cut)
                    
                self.train_predict[rotate_index] = output.detach().cpu().numpy()[0]
                self.test_predict[rotate_index] = test_output.detach().cpu().numpy()[0]
                pbar.update(1)
        self.evaluation()
