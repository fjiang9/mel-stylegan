import argparse
import os

import torch
from torch import optim
from torchvision import utils
from model import Generator
from tqdm import tqdm

import numpy as np
from model import Generator, Discriminator
from torch.nn import functional as F
from matplotlib import pyplot as plt
from audio import MelToAudio
from generate import generate_mel, mels_to_wav
import IPython.display as ipd
from ilo import LatentOptimizer

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda"


def load_model(load_path, device):
    checkpoint = torch.load(load_path)
    g_ema = Generator(checkpoint["args"].num_frames, checkpoint["args"].latent, checkpoint["args"].n_mlp,
                      channel_multiplier=checkpoint["args"].channel_multiplier).to(device)
    g_ema.load_state_dict(checkpoint["g_ema"])

    discriminator = Discriminator(checkpoint["args"].num_frames,
                                  channel_multiplier=checkpoint["args"].channel_multiplier).to(device)

    truncation = 1
    truncation_mean = 4096
    # if truncation < 1:
    #     with torch.no_grad():
    #         mean_latent = g_ema.mean_latent(truncation_mean)
    # else:
    #     mean_latent = None

    with torch.no_grad():
        mean_latent = g_ema.mean_latent(truncation_mean)

    return g_ema, mean_latent, discriminator


def inverse(x, generator, mean_latent, lr=0.01, iter=500, task='invert'):

    if task == 'separate':
        sample_z = torch.randn(2, generator.style_dim, device=device, requires_grad=True)
    else:
        sample_z = torch.randn(1, generator.style_dim, device=device, requires_grad=True)
    optimizer = optim.Adam([sample_z], lr=lr)
    generator.eval()
    res = []
    for i in tqdm(range(iter)):
        optimizer.zero_grad()
        x_hat, _ = generator([sample_z], truncation=1, truncation_latent=mean_latent,
                             input_is_latent=False)
        x_hat = x_hat.squeeze(1)

        if task == 'separate':
            loss = (0.2*torch.log10(10**(5*x_hat[0]-4) + 10**(5*x_hat[0]-4))+0.8 - x).pow(2).mean()
        else:
            loss = F.mse_loss(x_hat[0], x)
            # loss = (x_hat[0] - x).pow(2).sum()
            # loss = (10**(5*x_hat-4) - 10**(5*x-4)).pow(2).sum()

        res.append(loss.item())
        loss.backward()
        optimizer.step()
    # plt.figure(figsize=[12, 4])
    # plt.plot(res)
    # plt.show()
    return x_hat, res


def ilo_inverse(x, generator, device, lr=0.1, task='invert', search_space='W+', search_noise=False,
                steps='1000,500,500,500,500', start_layer=0, end_layer=5, mse_weight=1,
                discriminator=None, cls_alpha=0, mask=1, lpips_alpha=0, r_alpha=0.1):
    ilo_optimizer = LatentOptimizer(x, generator, device, steps=steps, lr=lr, task=task, search_space=search_space,
                                    search_noise=search_noise, project=True, start_layer=start_layer, end_layer=end_layer,
                                    discriminator=discriminator, cls_alpha=cls_alpha, mask=mask, mse_weight=mse_weight,
                                    lpips_alpha=lpips_alpha, r_alpha=r_alpha)
    inputs, z, images, res = ilo_optimizer.invert()
    return images, res, ilo_optimizer


def compare_inverse(x, x_hat, res=None):
    wav = mels_to_wav(x.unsqueeze(0))
    wav_hat = mels_to_wav(x_hat.unsqueeze(0))
    plt.figure(figsize=[8, 8])
    if res is not None:
        plt.subplot(2, 1, 1)
        plt.plot(res)
        plt.grid()
    plt.subplot(2, 2, 3)
    plt.imshow(x.detach().cpu(), origin='lower', cmap='magma')
    plt.subplot(2, 2, 4)
    plt.imshow(x_hat.detach().cpu(), origin='lower', cmap='magma')
    plt.show()
    ipd.display(ipd.Audio(wav, rate=16000))
    ipd.display(ipd.Audio(wav_hat, rate=16000))


def compare_mels(*mels, res=None):
    wavs = []
    for m in mels:
        wavs.append(mels_to_wav(m.unsqueeze(0)))
    plt.figure(figsize=[4*len(mels), 8])
    if res is not None:
        plt.subplot(2, 1, 1)
        plt.plot(res)
        plt.grid()
    for i, m in enumerate(mels):
        plt.subplot(2, len(mels), len(mels)+i+1)
        plt.imshow(m.detach().cpu(), origin='lower', cmap='magma')
    plt.show()
    for w in wavs:
        ipd.display(ipd.Audio(w, rate=16000))



