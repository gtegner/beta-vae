import torch
def KL_loss(mu, log_sigma):
    return 0.5 * torch.mean(1 - 2 * log_sigma + mu*mu + torch.exp(2 * log_sigma))