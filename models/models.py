import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal


class MNISTVAE(nn.Module):
    """Variational Auto Encoder for MNIST dataset based on the architecture given in https://arxiv.org/pdf/1312.6114.pdf

    Attributes:
        in_dim (int): the dimension of the input
        hidden_units (int): number of neurons on the hidden layer
        latent_dim (int): the dimension of the latent variable
    """
    def __init__(self, config: object):
        super().__init__()
        self.in_dim = config['in_dim']
        self.hidden_units = config['hidden_units']
        self.latent_dim = config['latent_dim']
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 2*self.latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_units),
            nn.Linear(self.hidden_units, 2*self.in_dim),
            nn.Sigmoid()
        )
        
    def elbo_loss(self, x:torch.Tensor):
        """Return the Evidence Lower BOund (ELBO) of the VAE

        Args:
            x (torch.Tensor): the input image

        Returns:
            torch.Tensor: the ELBO
        """
        params1 = self.encoder(x)
        mu_enc = params1[:,:self.latent_dim]
        log_sigma2_enc = params1[:,self.latent_dim:]
        B, D = params1.shape
        mu = torch.zeros(B, D//2)
        sigma = torch.ones(B, D//2)
        normal = Normal(loc=mu, scale=sigma)
        eps = normal.sample()
            
        z = mu_enc + torch.exp(0.5*log_sigma2_enc) * eps
        
        params2 = self.decoder(z)
        mu_dec = params2[:,:self.in_dim]
        log_sigma2_dec = params2[:,self.in_dim:]
            
        ## Negative Kullback-Leibler divergence term
        neg_kl_div = 0.5 * (1 + log_sigma2_enc - mu_enc**2 - torch.exp(log_sigma2_enc)).sum()
        
        ## Log-likelihood term
        sigma2 = torch.exp(log_sigma2_dec)
        densities = ((2*torch.pi*sigma2)**-0.5) * torch.exp(-(x - mu_dec)**2 / sigma2)
        log_likelihood = torch.log(densities).sum()
        
        ## ELBO
        elbo = neg_kl_div + log_likelihood
        
        
        return elbo
    
    @torch.no_grad()
    def encode(self, x:torch.Tensor):
        """Generate the latent variable from x

        Args:
            x (torch.Tensor): the input image

        Returns:
            torch.Tensor: the latent variable
        """
        params1 = self.encoder(x)
        mu_enc = params1[:,:self.latent_dim]
        log_sigma2_enc = params1[:,self.latent_dim:]
        B, D = params1.shape
        mu = torch.zeros(B, D//2)
        sigma = torch.ones(B, D//2)
        normal = Normal(loc=mu, scale=sigma)
        eps = normal.sample()
            
        z = mu_enc + torch.exp(0.5*log_sigma2_enc) * eps
        return z
    
    @torch.no_grad()
    def generate(self, z):
        """ Generate a new image from the latent variable

        Args:
            z (torch.Tensor): the latent variable

        Returns:
            torch.Tensor: the new generated image
        """
        params2 = self.decoder(z)
        mu_dec = params2[:,:self.in_dim]
        log_sigma2_dec = params2[:,self.in_dim:]
        normal = Normal(loc=mu_dec, scale=log_sigma2_dec)
        out = normal.sample()
        
        return out
    


class FreyFaceVAE(nn.Module):
    """Variational Auto Encoder for the Frey Face dataset based on the architecture given in https://arxiv.org/pdf/1312.6114.pdf

    Attributes:
        in_dim (int): the dimension of the input
        hidden_units (int): number of neurons on the hidden layer
        latent_dim (int): the dimension of the latent variable
    """
    def __init__(self, config):
        super().__init__()
        self.in_dim = config['in_dim']
        self.hidden_units = config['hidden_units']
        self.latent_dim = config['latent_dim']
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 2*self.latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_units),
            nn.Linear(self.hidden_units, 2*self.in_dim),
            nn.Sigmoid()
        )
        
    def elbo_loss(self, x:torch.Tensor):
        """Return the Evidence Lower BOund (ELBO) of the VAE

        Args:
            x (torch.Tensor): the input image

        Returns:
            torch.Tensor: the ELBO
        """
        params1 = self.encoder(x)
        mu_enc = params1[:,:self.latent_dim]
        log_sigma2_enc = params1[:,self.latent_dim:]
        B, D = params1.shape
        mu = torch.zeros(B, D//2)
        sigma = torch.ones(B, D//2)
        normal = Normal(loc=mu, scale=sigma)
        eps = normal.sample()
            
        z = mu_enc + torch.exp(0.5*log_sigma2_enc) * eps
        
        params2 = self.decoder(z)
        mu_dec = params2[:,:self.in_dim]
        log_sigma2_dec = params2[:,self.in_dim:]
            
        ## Negative Kullback-Leibler divergence term
        neg_kl_div = 0.5 * (1 + log_sigma2_enc - mu_enc**2 - torch.exp(log_sigma2_enc)).sum()
        
        ## Log-likelihood term
        sigma2 = torch.exp(log_sigma2_dec)
        densities = ((2*torch.pi*sigma2)**-0.5) * torch.exp(-(x - mu_dec)**2 / sigma2)
        log_likelihood = torch.log(densities).sum()
        
        ## ELBO
        elbo = neg_kl_div + log_likelihood
        
        
        return elbo
    
    @torch.no_grad()
    def encode(self, x:torch.Tensor):
        """Generate the latent variable from x

        Args:
            x (torch.Tensor): the input image

        Returns:
            torch.Tensor: the latent variable
        """
        params1 = self.encoder(x)
        mu_enc = params1[:,:self.latent_dim]
        log_sigma2_enc = params1[:,self.latent_dim:]
        B, D = params1.shape
        mu = torch.zeros(B, D//2)
        sigma = torch.ones(B, D//2)
        normal = Normal(loc=mu, scale=sigma)
        eps = normal.sample()
            
        z = mu_enc + torch.exp(0.5*log_sigma2_enc) * eps
        return z
    
    @torch.no_grad()
    def generate(self, z):
        """ Generate a new image from the latent variable

        Args:
            z (torch.Tensor): the latent variable

        Returns:
            torch.Tensor: the new generated image
        """
        params2 = self.decoder(z)
        mu_dec = params2[:,:self.in_dim]
        log_sigma2_dec = params2[:,self.in_dim:]
        normal = Normal(loc=mu_dec, scale=log_sigma2_dec)
        out = normal.sample()
        
        return out