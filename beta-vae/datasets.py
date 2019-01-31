import torch
import torch.utils.data
import numpy as np
import os

class SpritesDataset(torch.utils.data.Dataset):
    def __init__(self):
        root_dir = 'data'
        dataset_dir = 'dsprites-dataset'
        filename = 'data.npz'

        self.data = np.load(os.path.join(root_dir, dataset_dir,filename), encoding = 'latin1')

        self.imgs = self.data['imgs']
        self.latent_classes = self.data['latents_classes']
        self.latent_values = self.data['latents_values']
        self.metadata = self.data['metadata'][()]

        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

        del self.data

    def random_sample(self, size = 1):
        a = np.zeros((size, len(self.latents_sizes)))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            a[:, lat_i] = np.random.randint(lat_size, size = size)
        return a

    def latent_to_ix(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_images(self, size = 1):
        samples = self.random_sample(size)
        ix = self.latent_to_ix(samples)

        return self.imgs[ix]

    def cond_sampling(self, latent_factor, latent_value, size = 1):
        samples = self.random_sample(5000)
        samples = samples[np.where(samples[:, latent_factor] == latent_value)]

        return samples

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = torch.from_numpy(self.imgs[index]).float()
        latent_class = torch.from_numpy(self.latent_classes[index]).float()
        latent_value = torch.from_numpy(self.latent_values[index]).float()

        return img, latent_class, latent_value

    

if __name__ == '__main__':
    dataset = SpritesDataset('data', 'dsprites-dataset', 'data.npz')

    img = dataset.sample_images(1)
    print(img.shape)