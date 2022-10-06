import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        intermediate_dimensions = np.linspace(
            hidden_dim, latent_dim, hidden_layers+1, dtype=int)[0:-1]
        intermediate_dimensions = np.concatenate(
            ([input_dim + y_dim], intermediate_dimensions))
        for i, (in_size, out_size) in enumerate(zip(intermediate_dimensions[:-1], intermediate_dimensions[1:])):
            self.hidden.add_module(name='Linear_'+str(i),
                                   module=nn.Linear(in_size, out_size))
            #self.hidden.add_module(name='BN_'+str(i), module=nn.BatchNorm1d(out_size))
            self.hidden.add_module(name='Act_'+str(i), module=nn.ELU())
            #self.hidden.add_module(name='Drop_'+str(i), module=nn.Dropout(p=0.05, inplace=False))

        self.mu = nn.Linear(intermediate_dimensions[-1], latent_dim)
        self.var = nn.Linear(intermediate_dimensions[-1], latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        x = self.hidden(x)
        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(x)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(x)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''

    def __init__(self, latent_dim, hidden_dim, output_dim, hidden_layers, y_dim):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        intermediate_dimensions = np.linspace(
            latent_dim + y_dim, hidden_dim, hidden_layers+1, dtype=int)
        for i, (in_size, out_size) in enumerate(zip(intermediate_dimensions[:-1], intermediate_dimensions[1:])):
            self.hidden.add_module(name='Linear_'+str(i),
                                   module=nn.Linear(in_size, out_size))
            #self.hidden.add_module(name='BN_'+str(i), module=nn.BatchNorm1d(out_size))
            self.hidden.add_module(name='Act_'+str(i), module=nn.ELU())
            #self.hidden.add_module(name='Drop_'+str(i), module=nn.Dropout(p=0.05, inplace=False))

        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = self.hidden(x)

        generated_x = self.hidden_to_out(x)

        return generated_x


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input.
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim,
                               latent_dim, hidden_layers, y_dim)
        self.decoder = Decoder(latent_dim, hidden_dim,
                               input_dim, hidden_layers, y_dim)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

def kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)

    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl


def calculate_loss(x, reconstructed_x, mu, log_var, weight, mc_kl_loss):
    # reconstruction loss
    rcl = F.mse_loss(reconstructed_x, x, size_average=False)
    # kl divergence loss

    if mc_kl_loss == True:
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        kld = kl_divergence(z, mu, std).sum()
    else:
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return rcl, kld * weight