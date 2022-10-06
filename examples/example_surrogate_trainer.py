import torch, yaml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib, time, copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from structrepgen.models.models import ff_net
from structrepgen.utils.dotdict import dotdict
from structrepgen.utils.utils import torch_device_select

'''
Example of training a surrogate model based on the representation R extracted using Behler descriptors
The model is trained on MSE loss for y

Model used:
    fully connected feed-forward neural network
'''

class Trainer():
    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)

        # initialize
        self.create_data()
        self.initialize()

    def create_data(self):
        p = self.CONFIG.params

        data_x = pd.read_csv(self.CONFIG.data_x_path, header=None).values
        data_y = pd.read_csv(self.CONFIG.data_y_path, header=None).values

        # train/test split and create torch dataloader
        xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=self.CONFIG.split_ratio, random_state=p.seed)
        self.x_train = torch.tensor(xtrain, dtype=torch.float)
        self.y_train = torch.tensor(ytrain, dtype=torch.float)
        self.x_test = torch.tensor(xtest, dtype=torch.float)
        self.y_test = torch.tensor(ytest, dtype=torch.float)

        self.train_loader = DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=p.batch_size, shuffle=True, drop_last=False
        )

        self.test_loader = DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=p.batch_size, shuffle=False, drop_last=False
        )
    
    def initialize(self):
        p = self.CONFIG.params

        # create model
        self.model = ff_net(p.input_dim, p.hidden_dim, p.hidden_layers)
        self.model.to(self.device)
        print(self.model)

        # set up optimizer
        gamma = (p.final_decay)**(1./p.n_epochs) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)

    def train(self):
        p = self.CONFIG.params
        self.model.train()

        # loss of the epoch
        total_loss = 0.

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # forward
            y_pred = self.model(x)

            loss = F.mse_loss(y_pred, y, size_average=False)

            # backward
            loss.backward()
            total_loss += loss.item()

            # update the weights
            self.optimizer.step()
        
        return total_loss
    
    def test(self):
        p = self.CONFIG.params

        self.model.eval()

        # loss of the evaluation
        total_loss = 0.

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                y_pred = self.model(x)

                # loss
                loss = F.mse_loss(y_pred, y, size_average=False)
                total_loss += loss.item()
        
        return total_loss

    def run(self):
        p = self.CONFIG.params
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0

        for e in range(p.n_epochs):
            tic = time.time()

            train_loss = self.train()
            test_loss = self.test()

            train_loss /= len(self.x_train)
            test_loss /= len(self.x_test)

            self.scheduler.step()
            lr = self.scheduler.optimizer.param_groups[0]["lr"]

            if best_test_loss > test_loss:
                best_epoch = e
                best_test_loss = test_loss
                best_train_loss = train_loss
                model_best = copy.deepcopy(self.model)
            
            elapsed_time = time.time() - tic
            epoch_out = f'Epoch {e:04d}, Train: {train_loss:.4f}, Test: {test_loss:.4f}, LR: {lr:.5f}, Time/Epoch (s): {elapsed_time:.3f}'
            if e % p.verbosity == 0:
                print(epoch_out)
        
        torch.save(model_best, self.CONFIG.ff_path)
        return best_epoch, best_train_loss, best_test_loss

if __name__ == "__main__":
    # load parameters from yaml file
    stream = open('./configs/example/example_ff_trainer.yaml')
    CONFIG = yaml.safe_load(stream)
    stream.close()
    CONFIG = dotdict(CONFIG)

    trainer = Trainer(CONFIG)
    trainer.run()