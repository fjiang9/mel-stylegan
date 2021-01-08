from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn import functional as F

import numpy as np
import os
import torch
import torchvision.transforms as transforms
import glob


class SpokenDigits(Dataset):
    """Melspectrogram dataset"""

    def __init__(self, mel_dir, num_mels, transform=None):
        self.mel_dir = mel_dir
        self.num_mels = num_mels
        self.transform = transform
        self.files = os.listdir(mel_dir)
        self.labels_map = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
        }

    def __getitem__(self, idx):
        specgram = self._load_melspecgram(self.files[idx])
        label = self._get_label(self.files[idx].split("_")[0].lower())
        if self.transform:
            specgram = self.transform(specgram)
        specgram = specgram.type(torch.FloatTensor)
        label = torch.LongTensor([label])
        return specgram #, label

    def __len__(self):
        return len(self.files)

    def _get_label(self, label_str):
        label = self.labels_map.get(label_str)
        if label is None:
            raise Exception("Couldn't find label for {}".format(label_str))
        return label

    def _load_melspecgram(self, filename):
        spec_obj = np.load(os.path.join(self.mel_dir, filename))
        return spec_obj

    def show_mel_spectrogram(self, filename):
        spec = self._load_melspecgram(filename)
        plt.imshow(spec.T)
        plt.show()


def spoken_digits_loader(mel_path, batch_size, num_frames, num_mels):
    composed = transforms.Compose(
        [
            TimePadding(length=num_frames),
            ToTensor(dims=3),
            ResizeTensor(num_mels, num_frames)
        ]
    )
    dataset = SpokenDigits(mel_dir=mel_path, num_mels=num_mels, transform=composed)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


class LibriSpeech(Dataset):
    def __init__(self, data_path, subset='train-clean-100', num_frames=128):
        '''
        :param data_path: path the mel features
        :param subset: can be 'train-clean-100', 'train-clean-360', '*'
        :param num_frames:
        '''
        self.files = glob.glob(os.path.join(data_path, subset, '*.npy'))
        self.num_frames = num_frames

    def __len__(self):
        return len(self.files)

    def fit_size(self, x, thr, idx):
        if x.shape[1] < self.num_frames:
            lp = np.random.randint(self.num_frames-x.shape[1])
            x_ = np.pad(x, ((0, 0), (lp, self.num_frames-x.shape[1]-lp)),
                       'constant', constant_values=((0, 0), (0, 0)))
        elif x.shape[1] > self.num_frames:
            cnt = 0
            while cnt < 20:
                st = np.random.randint(x.shape[1]-self.num_frames)
                x_ = x[:, st:st+self.num_frames]
                if x_.sum(axis=0).mean() > thr:
                    break
                cnt += 1
            if cnt >=20:
                print('Warning: {} could be silence'.format(self.files[idx]))
        else:
            x_ = x
        return x_

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        x = self.fit_size(x, thr=21, idx=idx)
        x = torch.from_numpy(x).float().unsqueeze(0)
        return x




class Nancy16(Dataset):
    """Melspectrogram dataset"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __getitem__(self, idx):
        specgram = self._load_melspecgram(self.files[idx])

        if self.transform:
            specgram = self.transform(specgram)

        return specgram

    def __len__(self):
        return len(self.files)

    def _load_melspecgram(self, filename):
        spec_obj = np.load(os.path.join(self.root_dir, filename))
        return spec_obj

    def show_mel_spectrogram(self, filename):
        spec = self._load_melspecgram(filename)
        plt.imshow(spec.T)
        plt.show()

class Nancy16WithLabs(Dataset):
    """Melspectrogram dataset with corresponding labs data"""

    def __init__(self, mel_dir, num_mels, lab_tensor_dir, mel_transform=None, lab_transform=None):
        self.mel_dir = mel_dir
        self.num_mels = num_mels
        self.lab_tensor_dir = lab_tensor_dir
        self.mel_transform = mel_transform
        self.lab_transform = lab_transform
        self.mel_files = os.listdir(self.mel_dir)
        self.fnames = list(map(lambda x: x.rstrip(".npy"), self.mel_files))
        self.lab_tensor_files = [f + ".pt" for f in self.fnames]

    def lab_len_stats(self):
        lab_lens = []
        for lab in self.lab_tensor_files:
            lab_tensor = torch.load(os.path.join(self.lab_tensor_dir, lab))
            lab_lens.append((lab_tensor != 0).sum().item())

        return np.mean(lab_lens), np.std(lab_lens)

    def __getitem__(self, idx):
        specgram = self._load_melspecgram(self.mel_files[idx])
        lab_tensor = self._load_lab_tensor(self.lab_tensor_files[idx])
        lab_length = (lab_tensor != 0).sum()

        if self.mel_transform:
            specgram = self.mel_transform(specgram)

        if self.lab_transform:
            lab_tensor = self.lab_transform(lab_tensor)

        return self._asfloat(specgram), (lab_tensor, lab_length)

    def __len__(self):
        return len(self.mel_files)

    def _load_melspecgram(self, filename):
        spec_obj = np.load(os.path.join(self.mel_dir, filename))
        return spec_obj

    def _load_lab_tensor(self, filename):
        lab_tensor = torch.load(os.path.join(self.lab_tensor_dir, filename))
        return lab_tensor

    def show_mel_spectrogram(self, filename):
        spec = self._load_melspecgram(filename)
        plt.imshow(spec.T)
        plt.show()

    @staticmethod
    def _asfloat(tensor):
        tensor = tensor.type(torch.FloatTensor)
        return tensor


class FixedCrop:
    """Crops the melspectorgram at fixed point on time axis"""

    def __init__(self, start_ix, height):
        self.start_ix = start_ix
        self.end_ix = height - start_ix + 1

    def __call__(self, sample):
        sample = sample[:, self.start_ix:self.end_ix]
        return sample

class ToTensor:
    """Convert PIL-images/ndarrays in sample to Tensors."""

    def __init__(self, dims, ):
        self.dims = dims

    def __call__(self, sample):
        if isinstance(sample, np.ndarray):

            if self.dims == 3:
                out = torch.from_numpy(sample[np.newaxis, :, :])
            elif self.dims == 2:
                out = torch.from_numpy(sample)
            return out

        elif isinstance(sample, Image.Image):
            out = torch.ByteTensor(torch.ByteStorage.from_buffer(sample.tobytes()))

        if self.dims == 3:
            # put it from HWC to CHW format
            out = out.view(sample.size[1], sample.size[0], 1)
            out = out.transpose(0, 1).transpose(0, 2).contiguous()
        elif self.dims == 2:
            # put it from HWC to HW format
            out = out.view(sample.size[1], sample.size[0])
            out = out.transpose(0, 1).contiguous()

        if isinstance(out, torch.ByteTensor):
            return out.float().div(255)
        return out

class NormalizeTensor:
    """Normalize tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std

class TimePadding:
    """Zero pads the time axis of mel to match desired length"""

    def __init__(self, length):
        self.length = length

    def __call__(self, sample, value=0):
        num_pads = max(0, self.length - sample.shape[1])
        left_padding = num_pads // 2
        right_padding = num_pads - left_padding
        padded = np.pad(sample, [(0, 0), (left_padding, right_padding)], "constant", constant_values=value)
        return padded

class ResizeTensor:

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, tensor):
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))
        tensor =  F.interpolate(tensor, size=(self.height, self.width), mode='bilinear')
        return tensor.view(tensor.size(1), tensor.size(2), tensor.size(3))


if __name__ == "__main__":
    num_mels = 128
    num_frames = 128
    batch_size = 4
    melspec_path = "/home/neil/fei/data/sc09_features/train/melspec_{num_mels}".format(num_mels=num_mels)
    dataset = spoken_digits_loader(melspec_path, batch_size, num_frames, num_mels)
    sample_size = len(os.listdir(melspec_path))
    loader = iter(dataset)
    real_image, real_labels = next(loader)
    print(real_image.shape)
    print(real_labels)




