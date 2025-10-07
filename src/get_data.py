import numpy as np
import torch
from sklearn.datasets import make_moons
from PIL import Image

def inf_train_gen(img_path, data_size):
    def gen_data_from_img(image_mask, train_data_size):
        def sample_data(train_data_size):
            inds = np.random.choice(
                int(probs.shape[0]), int(train_data_size), p=probs)
            m = means[inds] 
            samples = np.random.randn(*m.shape) * std + m 
            samples = torch.tensor(samples)
            return samples
        img = image_mask
        h, w = img.shape
        xx = np.linspace(-4, 4, w)
        yy = np.linspace(-4, 4, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        means = np.concatenate([xx, yy], 1) # (h*w, 2)
        img = img.max() - img
        probs = img.reshape(-1) / img.sum() 
        std = np.array([8 / w / 2, 8 / h / 2])
        full_data = sample_data(train_data_size)
        return full_data
    image_mask = np.array(Image.open(img_path).rotate(
        180).transpose(0).convert('L'))
    dataset = gen_data_from_img(image_mask, data_size)
    return dataset.float()

def gen_moons(num_samples: int, device="cpu", seed=42):
    xraw, _ = make_moons(noise=0.05, n_samples=num_samples, random_state=seed)
    # Scale to same domain as checkerboard
    mean = xraw.mean(axis=0)
    std = xraw.std(axis=0) / np.array([np.sqrt(4), np.sqrt(5)])
    xraw = (xraw - mean) / std
    xraw = torch.from_numpy(xraw).float().to(device)
    return xraw
