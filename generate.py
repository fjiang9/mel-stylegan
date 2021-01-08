import argparse
import os

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

import numpy as np
from model import Generator
from matplotlib import pyplot as plt
from audio import MelToAudio
import IPython.display as ipd

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda"



def generate_mel(load_path, num_samples=32, device="cuda"):
    checkpoint = torch.load(load_path)
    g_ema = Generator(checkpoint["args"].num_frames, checkpoint["args"].latent, checkpoint["args"].n_mlp,
                      channel_multiplier=checkpoint["args"].channel_multiplier).to(device)
    g_ema.load_state_dict(checkpoint["g_ema"])

    truncation = 1
    truncation_mean = 4096
    if truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(truncation_mean)
    else:
        mean_latent = None

    sample_z = torch.randn(num_samples, checkpoint["args"].latent, device=device)
    g_ema.eval()
    with torch.no_grad():
        sample, _ = g_ema([sample_z], truncation=1, truncation_latent=mean_latent)
    samples = sample.squeeze(1).detach().cpu()

    plt.figure(figsize=[16, 2*int(np.ceil(num_samples/8))])
    for i in range(samples.shape[0]):
        plt.subplot(int(np.ceil(num_samples/8)), 8, i+1)
        plt.imshow(samples[i], origin='lower', cmap='magma')
    plt.show()
    return samples

def save_images(mels, path):
    plt.figure(figsize=[16, 2 * int(np.ceil(mels.shape[0] / 8))])
    for i in range(mels.shape[0]):
        plt.subplot(int(np.ceil(mels.shape[0] / 8)), 8, i + 1)
        plt.imshow(mels[i], origin='lower', cmap='magma')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.savefig(path, dpi=50)
    plt.close()


def mels_to_wav(mels):
    m2a = MelToAudio(num_mels=mels.size(1))
    wavs = []
    for mel in tqdm(mels):
        wav = m2a.inv_melspectrogram(mel.cpu().data.numpy())
        wavs.append(wav)
    return wavs


def play_audio(mels):
    wavs = mels_to_wav(mels)
    for wav in wavs:
        ipd.display(ipd.Audio(wav, rate=16000))


def mels_to_audio(images, labels, path):
    m2a = MelToAudio(num_mels=images.size(1))
    if not os.path.exists(path):
        os.mkdir(path)

    for img, label in tqdm(zip(images, labels)):
        audio = m2a.inv_melspectrogram(img.cpu().data.numpy())
        idx = str(_get_sample_idx(path))
        m2a.save_wav(audio, os.path.join(path, ".".join([str(label.item()), idx, "wav"])))


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=128, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint/030000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.num_frames, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
