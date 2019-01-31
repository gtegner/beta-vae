import argparse
import numpy as np


from models import betaVAE
from datasets import SpritesDataset

import torch
from torch.utils.data import DataLoader
from utils import KL_loss
from tqdm import tqdm


parser = argparse.ArgumentParser()


if __name__ == '__main__':
    dataset = SpritesDataset()
    train_loader = DataLoader(dataset, batch_size = 100, num_workers = 2, shuffle = True)

    img, _, _ = next(iter(train_loader))

    input_dim = np.prod(img[1].size())


    model = betaVAE(input_dim, 20)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    epochs = 100
    
    beta = 1
    eps = 0.01
    recon_loss = torch.nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(epochs)):
        
        for i, (img, _, _) in enumerate(train_loader):
            img = img.view(img.size(0), -1)
            x_recon, mu, log_sigma = model(img)
            
            loss_ = recon_loss(x_recon, img)
            kl_loss = KL_loss(mu, log_sigma)
            loss = loss_ + beta * torch.abs(kl_loss - eps)
            
            loss.backward()
            opt.step()
            opt.zero_grad()

            if i > 10:
                break
        
        if epoch % 1 == 0:
            print("Epoch - {}, Loss - {}".format(epoch,loss))


            
            
