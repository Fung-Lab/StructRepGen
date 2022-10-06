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

from structrepgen.models.CVAE import *
from structrepgen.utils.dotdict import dotdict
from structrepgen.utils.utils import torch_device_select

'''
Example of training a CVAE based on the representation R extracted using Behler descriptors
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

        # scale
        scaler = MinMaxScaler()
        data_x = scaler.fit_transform(data_x)
        joblib.dump(scaler, self.CONFIG.scaler_path)

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
        self.model = CVAE(p.input_dim, p.hidden_dim, p.latent_dim, p.hidden_layers, p.y_dim)
        self.model.to(self.device)
        print(self.model)

        # set up optimizer
        gamma = (p.final_decay)**(1./p.n_epochs) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)

    def train(self):
        p = self.CONFIG.params
        self.model.train()

        # loss of the peoch
        rcl_loss = 0.
        kld_loss = 0.

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # forward
            reconstructed_x, z_mu, z_var = self.model(x, y)

            rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, p.kl_weight, p.mc_kl_loss)

            # backward
            combined_loss = rcl + kld
            combined_loss.backward()
            rcl_loss += rcl.item()
            kld_loss += kld.item()

            # update the weights
            self.optimizer.step()
        
        return rcl_loss, kld_loss
    
    def test(self):
        p = self.CONFIG.params

        self.model.eval()

        # loss of the evaluation
        rcl_loss = 0.
        kld_loss = 0.

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                reconstructed_x, z_mu, z_var = self.model(x, y)

                # loss
                rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, p.kl_weight, p.mc_kl_loss)
                rcl_loss += rcl.item()
                kld_loss += kld.item()
        
        return rcl_loss, kld_loss

    def run(self):
        p = self.CONFIG.params
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0

        for e in range(p.n_epochs):
            tic = time.time()

            rcl_train_loss, kld_train_loss = self.train()
            rcl_test_loss, kld_test_loss = self.test()

            rcl_train_loss /= len(self.x_train)
            kld_train_loss /= len(self.x_train)
            train_loss = rcl_train_loss + kld_train_loss
            rcl_test_loss /= len(self.x_test)
            kld_test_loss /= len(self.x_test)
            test_loss = rcl_test_loss + kld_test_loss

            self.scheduler.step()
            lr = self.scheduler.optimizer.param_groups[0]["lr"]

            if best_test_loss > test_loss:
                best_epoch = e
                best_test_loss = test_loss
                best_train_loss = train_loss
                model_best = copy.deepcopy(self.model)
            
            elapsed_time = time.time() - tic
            epoch_out = f'Epoch {e:04d}, Train RCL: {rcl_train_loss:.3f}, Train KLD: {kld_train_loss:.3f}, Train: {train_loss:.3f}, Test RLC: {rcl_test_loss:.3f}, Test KLD: {kld_test_loss:.3f}, Test: {test_loss:.3f}, LR: {lr:.5f}, Time/Epoch (s): {elapsed_time:.3f}'
            if e % p.verbosity == 0:
                print(epoch_out)
        
        torch.save(model_best, self.CONFIG.model_path)
        return best_epoch, best_train_loss, best_test_loss

if __name__ == "__main__":
    # load parameters from yaml file
    stream = open('./configs/example/example_cvae_trainer.yaml')
    CONFIG = yaml.safe_load(stream)
    stream.close()
    CONFIG = dotdict(CONFIG)

    trainer = Trainer(CONFIG)
    trainer.run()