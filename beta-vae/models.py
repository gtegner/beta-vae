import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(encoder, self).__init__()

        self.layers = nn.Sequential(nn.Linear(input_dim, 100), nn.ReLU(), nn.Linear(100, output_dim))
        self.enc_mu = nn.Linear(output_dim, latent_dim)
        self.enc_logsigma = nn.Linear(output_dim, latent_dim)
        
    def reparam(self, mu, log_sigma):
        eps = torch.randn_like(log_sigma)

        return mu + torch.exp(log_sigma) * eps


    def forward(self, x):
        h = self.layers(x)

        mu = self.enc_mu(h)
        log_sigma = self.enc_logsigma(h)

        z = self.reparam(mu, log_sigma)
        return z, mu, log_sigma

class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(decoder, self).__init__()

        self.layers = nn.Sequential(nn.Linear(latent_dim, 100), nn.ReLU(), nn.Linear(100, input_dim))

    def forward(self, x):
        return self.layers(x)


class betaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(betaVAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, 100)
        self.decoder = decoder(input_dim, latent_dim)
    
    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_recon = self.decoder(z)

        return x_recon, mu, log_sigma

