import torch, joblib, os
import numpy as np

from structrepgen.models.models import *
from structrepgen.models.CVAE import *
from structrepgen.utils.utils import torch_device_select

class Generator:
    '''
    Generator class for generating representation R from decoder

    # TODO
        - make this generic such that different decoders can be used

        Parameters:
                CONFIG (dict): model configurations for generation 

            Returns:
                NIL
    '''
    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)

        # load models
        if self.device == 'cpu':
            self.model = torch.load(CONFIG.model_path, map_location=torch.device(self.device))
            self.model_ff = torch.load(CONFIG.ff_model_path, map_location=torch.device(self.device))
        else:
            self.model = torch.load(CONFIG.model_path)
            self.model_ff = torch.load(CONFIG.ff_model_path)
        
        self.model.to(self.device)
        self.model_ff.to(self.device)
        self.model.eval()
        self.model_ff.eval()

        self.rev_x = torch.zeros((len(self.CONFIG.targets), self.CONFIG.num_z, self.CONFIG.params.input_dim)).to(self.device)

    def generate(self):
        '''
        Generate representation R using decoder by randomly sampling the latent space
        Output csv saved in folder defined in yaml file

        Parameters:
                NIL

            Returns:
                NIL
        '''
        p = self.CONFIG.params
        for count, y0 in enumerate(self.CONFIG.targets):
            for i in range(self.CONFIG.num_z):
                z = torch.randn(1, p.latent_dim).to(self.device)
                y = torch.tensor([[y0]]).to(self.device)

                z = torch.cat((z, y), dim=1)

                reconstructed_x = self.model.decoder(z)
                self.rev_x[count, i, :] = reconstructed_x
            
            fname = self.CONFIG.save_path + 'gen_samps_cvae_' + str(y0) + '.csv'
            np.savetxt(fname, self.rev_x[count, :, :].cpu().data.numpy(), fmt='%.6f', delimiter=',')
    
    def range_check(self):
        '''
        Check percentage of generated structures that have y value within +- self.CONFIG.delta of target y value
        '''
        scaler = joblib.load(self.CONFIG.scaler_path)
        rev_x_scaled = scaler.inverse_transform(self.rev_x.reshape(-1, self.rev_x.shape[2]).cpu().data.numpy())
        rev_x = torch.tensor(rev_x_scaled).to(self.device).reshape(self.rev_x.shape)

        ratios = 0
        avg_diff = 0

        for count, y0 in enumerate(self.CONFIG.targets):
            y1 = self.model_ff(rev_x[count, :,:])

            indices = torch.where((y1 > y0 - self.CONFIG.delta) & (y1 < y0 + self.CONFIG.delta))[0]
            ratio = len(indices)/len(y1) * 100
            ratios += ratio
            rev_x_out = rev_x[count, indices, :]

            minn, maxx = min(self.CONFIG.targets), max(self.CONFIG.targets)
            y_indices = torch.where((y1 > minn) & (y1 < maxx))[0]
            average = torch.mean(y1[y_indices]).item()
            avg_diff += abs(y0-average)

            out = f'Target: {y0:.2f}, Average value: {average:.2f}, Percent of samples within range: {ratio:.2f}'
            print(out)
            
            fname = self.CONFIG.save_path + 'gen_samps_x_' + str(y0) + '.csv'
            np.savetxt(fname, rev_x_out.cpu().data.numpy(), fmt='%.6f', delimiter=',')
        
        ratios = ratios/len(self.CONFIG.targets)
        avg_diff = avg_diff/len(self.CONFIG.targets)
        avg_out = f'Average difference: {avg_diff:.2f}, Average percent: {ratios:.2f}'
        print(avg_out)

        return avg_diff, ratios