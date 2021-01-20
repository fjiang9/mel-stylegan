# changed from https://github.com/giannisdaras/ilo/blob/master/ilo_stylegan.py
import torchvision
import numpy as np
import math
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import lpips
from model import Generator
torch.set_printoptions(precision=5)
from torch import nn
from torch.nn import functional as F
# from robustness import imagenet_models
# from collections import OrderedDict
# from utils import *
from torch.optim import Optimizer

def get_transformation(image_size):
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


class MappingProxy(nn.Module):
    def __init__(self,gaussian_ft):
        super(MappingProxy,self).__init__()
        self.mean = gaussian_ft["mean"]
        self.std = gaussian_ft["std"]
        self.lrelu = torch.nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.lrelu(self.std * x + self.mean)
        return x

def loss_geocross(latent, latent_dim):
        if latent.size() == (1, latent_dim):
            return 0
        else:
            num_latents  = latent.size()[1]
            X = latent.view(-1, 1, num_latents, latent_dim)
            Y = latent.view(-1, num_latents, 1, latent_dim)
            A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
            B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
            D = 2 * torch.atan2(A, B)
            D = ((D.pow(2) * latent_dim).mean((1, 2)) / 8.).mean()
            return D

class SphericalOptimizer():
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt() for param in params}
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])


# class SphericalOptimizer(Optimizer):
#     def __init__(self, optimizer, params, **kwargs):
#         self.opt = optimizer(params, **kwargs)
#         self.params = params
#         with torch.no_grad():
#             self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}
#
#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = self.opt.step(closure)
#         for param in self.params:
#             param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
#             param.mul_(self.radii[param])
#
#         return loss


class LatentOptimizer(torch.nn.Module):
    def __init__(self, x, gma, device, lr=0.1, steps='1000', task='invert',
                 search_space='W+', search_noise=True, project=True,
                 start_layer=0, end_layer=5, discriminator=None,
                 cls_alpha=0, mask=1, mse_weight=1, lpips_alpha=0, r_alpha=0.1):
        """

        :param x:
        :param gma:
        :param device:
        :param lr:
        :param steps:
        :param task:
        :param search_space: W, W+, Z, Z+
        :param search_noise:
        :param project:
        :param start_layer:
        :param end_layer:
        :param discriminator:
        :param cls_alpha:
        :param mask:
        :param mse_weight:
        :param lpips_alpha:
        :param r_alpha:
        """
        super().__init__()
        # self.config = config
        # if config['image_size'][0] != config['image_size'][1]:
        #     raise Exception('Non-square images are not supported yet.')
 
        self.task = task
        self.search_space = search_space
        self.search_noise = search_noise
        self.project = project
        self.steps = steps
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.dead_zone_linear = 0
        self.dead_zone_linear_alpha = 0.1
        self.device = device
        self.geocross_alpha = 0.1
        self.cls_alpha = cls_alpha
        self.lpips_alpha=lpips_alpha
        self.r_alpha = r_alpha
        self.mask = mask
        self.mse_weight = mse_weight

        self.layer_in = None
        self.best = None
        self.skip = None
        self.lr = lr
        self.lr_record = []
        self.current_step = 0

        self.original_imgs = x.to(device)

        self.discriminator = discriminator
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(device)

        if self.task == 'separate':
            bs = self.original_imgs.shape[0] * 2
        else:
            bs = self.original_imgs.shape[0]

        # self.downsampler_1024_256 = BicubicDownSample(4)
        # self.downsampler_1024_image = BicubicDownSample(1024 // config['image_size'][0])
        # self.downsampler_image_256 = BicubicDownSample(config['image_size'][0] // 256)
        # Load models and pre-trained weights
        self.gen = gma.to(device)
        self.gen.start_layer = start_layer
        self.gen.end_layer = end_layer


        for p in self.gen.parameters():
            p.requires_grad = False

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.plrelu = torch.nn.LeakyReLU(negative_slope=5)

        # if self.verbose: print("\tRunning Mapping Network")
        with torch.no_grad():
            # torch.manual_seed(0)
            # latent = torch.randn((1000000, 512), dtype=torch.float32, device="cuda")
            # latent_out = torch.nn.LeakyReLU(5)(self.gen.style(latent))

            latent_p = self.plrelu(self.gen.style(torch.randn((500000, 512), dtype=torch.float32, device="cuda"))).double()
            self.mu = latent_p.mean(dim=0, keepdim=True)
            self.Sigma = (latent_p - self.mu).T @ (latent_p - self.mu) / latent_p.shape[0]
            d, V = torch.symeig(self.Sigma, eigenvectors=True)
            # small eigenvalues do not get overamplified.
            D = torch.diag(1. / torch.sqrt(d + 1e-18))
            # whitening matrix
            # W = np.dot(np.dot(V, D), V.T) # ZCA whitening
            self.W = (V @ D).float()  # PCA whitening
            self.W_inv = torch.inverse(self.W)

            latent_p = latent_p.float()
            self.mu = self.mu.float().unsqueeze(0).to(device)
            self.Sigma = self.Sigma.float().to(device)
            self.gaussian_fit = {"mean": latent_p.mean(0).to(device), "std": latent_p.std(0).to(device)}

            del latent_p
            torch.cuda.empty_cache()


        # self.mpl = MappingProxy(torch.load('gaussian_fit.pt'))

        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg",
                                            use_gpu=device.startswith("cuda"))


        # # load a classifier
        # self.cls = imagenet_models.resnet50()
        # state_dict = torch.load('imagenet_l2_3_0.pt')['model']
        # new_dict = OrderedDict()
        #
        # for key in state_dict.keys():
        #     if 'module.model' in key:
        #         new_dict[key[13:]] = state_dict[key]
        #
        # self.cls.load_state_dict(new_dict)
        # self.cls.to(config['device'])

        # initialization
        # self.scalar = torch.ones(bs, requires_grad=True).to(device)
        self.scalar = torch.ones(
            (bs, 1, 1, 1),
            dtype=torch.float,
            requires_grad=True, device='cuda')
        if start_layer == 0:
            noises_single = self.gen.make_noise(bs)
            self.noises = []
            for noise in noises_single:
                self.noises.append(noise.normal_())
            if self.search_space == 'W':
                # # self.latent_z = torch.randn(
                # #             (bs, self.gen.n_latent, self.gen.style_dim),
                # #             dtype=torch.float,
                # #             requires_grad=True, device='cuda')
                # with torch.no_grad():
                #     self.latent_z = self.gen.style(F.normalize(torch.randn(bs, self.gen.n_latent, self.gen.style_dim), p=2, dim=2).to(device)) # random w
                #     # self.latent_z = self.gen.style(F.normalize(torch.randn(bs, 1, self.gen.style_dim), p=2, dim=2).repeat(1, self.gen.n_latent, 1).to(device))  # random w
                #     # self.latent_z = self.gen.mean_latent(16384).unsqueeze(1).repeat(bs, self.gen.n_latent, 1).to(device) # mean w
                # self.latent_z.requires_grad = True
                # Generate latent tensor
                self.latent = torch.randn((bs, 1, self.gen.style_dim), dtype=torch.float, requires_grad=True, device='cuda')
            elif self.search_space == 'W+':
                self.latent = torch.randn((bs, self.gen.n_latent, self.gen.style_dim), dtype=torch.float, requires_grad=True, device='cuda')
                # with torch.no_grad():
                #     self.latent = self.gen.style(torch.randn(bs, self.gen.n_latent, self.gen.style_dim).to(device)) # random w
                # self.latent.requires_grad = True
            elif self.search_space == 'Z':
                self.latent = torch.randn(
                    (bs, 1, self.gen.style_dim),
                    dtype=torch.float,
                    requires_grad=True, device='cuda')
            elif self.search_space == 'Z+':
                # self.latent_z = torch.randn(
                #             (bs, self.gen.style_dim),
                #             dtype=torch.float,
                #             requires_grad=True, device='cuda')
                self.latent_z = torch.randn(
                    (bs, self.gen.n_latent, self.gen.style_dim),
                    dtype=torch.float,
                    requires_grad=True, device='cuda')
                self.latent_w = self.gen.style(self.latent_z)
            else:
                raise ValueError("searching_space incorrect")

            self.gen_outs = [None]
        else:
            # restore noises
            self.noises = torch.load(config['saved_noises'][0])
            self.latent_z = torch.load(config['saved_noises'][1]).to(config['device'])
            self.gen_outs = torch.load(config['saved_noises'][2])
            self.latent_z.requires_grad = True

    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp


    def z_to_w(self, latent_z):
        if self.search_space == "W":
            latent_w = self.lrelu(latent_z * self.gaussian_fit["std"] +
                                  self.gaussian_fit["mean"]).repeat(1, self.gen.n_latent, 1)
            # latent_w = self.latent_z
            # latent_w = self.mpl(self.latent_z)
            # latent_w = self.lrelu(self.latent_z * self.gaussian_fit["std"] + self.gaussian_fit["mean"])
        elif self.search_space == "W+":
            # latent_w = self.lrelu(self.latent)
            latent_w = self.lrelu(self.latent * self.gaussian_fit["std"] + self.gaussian_fit["mean"])
            # latent_w = self.latent
            # latent_w = self.lrelu((self.W_inv.unsqueeze(0).bmm(latent_z.permute(0, 2, 1))).permute(0, 2, 1) + self.mu)
        else:
            latent_w = self.gen.style(latent_z)
        return latent_w


    def invert_(self, start_layer, noise_list, steps, res, verbose=False, project=False):
        # noise_list containts the indices of nodes that we will be optimizing over
        for i in range(len(self.noises)):
            if i in noise_list:
                self.noises[i].requires_grad = True
            else:
                self.noises[i].requires_grad = False

        with torch.no_grad():
            if start_layer == 0:
                var_list = [self.latent] #+ [self.scalar]
            else:
                # intermediate_out = torch.ones(self.gen_outs[-1].shape, device=self.gen_outs[-1].device) * self.gen_outs[-1]
                # intermediate_out.requires_grad = True
                var_list = [self.latent] + [self.gen_outs[-1]] #+ [self.scalar]
            if self.search_noise:
                var_list += self.noises

            # set network that we will be optimizing over
            self.gen.start_layer = start_layer
            self.gen.end_layer = self.end_layer

        optimizer = optim.Adam(var_list, lr=self.lr)
        ps = SphericalOptimizer([self.latent] + self.noises)
        pbar = tqdm(range(steps))
        self.current_step += steps

        if self.task == 'inpaint':
            curr_shape = self.original_imgs.shape
            # curr_shape = [curr_shape[0]] + self.config['image_size']
            mask = torch.ones(curr_shape, device=self.device)
            bs, x, y = torch.where(self.original_imgs.sum(dim=1) == -3)
            mask[bs, :, x, y] = 0
        mse_min = np.inf

        lr_func = lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10)
        patient = 1000
        no_improve = 0
        for i in pbar:
            t = i / steps
            # lr = self.get_lr(t, self.lr) #* (0.1**(start_layer))
            lr = self.lr * (0.1**(start_layer))
            optimizer.param_groups[0]["lr"] = lr
            self.lr_record.append(lr)

            latent_w = self.z_to_w(self.latent)
            img_gen, _ = self.gen([latent_w],
                                  input_is_latent=True,
                                  noise=self.noises,
                                  layer_in=self.gen_outs[-1],)
            img_gen = img_gen * self.scalar
            batch, channel, height, width = img_gen.shape
            factor = height // 256
            # calculate loss
            if self.task == 'inpaint':
                # downsample generared images
                downsampled = self.downsampler_1024_image(img_gen)
                # mask
                masked = downsampled * mask
                # compute loss
                diff = torch.abs(masked - self.original_imgs) - self.dead_zone_linear_alpha
                dead_zone_linear_loss = torch.max(torch.zeros(diff.shape, device=diff.device), diff).mean()
                mse_loss = F.mse_loss(masked, self.original_imgs)
                if self.config['lpips_method'] == 'mask':
                    p_loss = self.percept(self.downsampler_image_256(masked),
                                          self.downsampler_image_256(self.original_imgs)).mean()
                elif self.config['lpips_method'] == 'fill':
                    filled = mask * self.original_imgs + (1 - mask) * downsampled
                    p_loss = self.percept(self.downsampler_1024_256(img_gen), self.downsampler_image_256(filled)).mean()
                else:
                    raise NotImplementdError('LPIPS policy not implemented')
            elif self.task == 'invert':
                # diff = torch.abs(img_gen - self.original_imgs) - self.dead_zone_linear_alpha
                # dead_zone_linear_loss = torch.max(torch.zeros(diff.shape, device=diff.device), diff).mean()
                # print(img_gen.shape)
                # print(self.original_imgs.shape)

                # mse_loss = F.mse_loss(img_gen*self.mask, self.original_imgs*self.mask)
                mse_loss = ((img_gen - self.original_imgs)*self.mask*self.mse_weight).pow(2).mean()
                # mse_loss = (10**(5*img_gen-4) - 10**(5*self.original_imgs-4)).pow(2).sum()

                filled = self.mask * self.original_imgs + (1 - self.mask) * img_gen
                p_loss = self.percept(img_gen, filled).mean()

                # p_loss = self.percept(img_gen, self.original_imgs).mean()

                # p_loss = self.percept(img_gen, self.resized_imgs).mean()
            elif self.task == 'separate':
                # # mse_loss = (10**(5*img_gen[0]-4) + 10**(5*img_gen[1]-4) - 10**(5*self.original_imgs-4)).pow(2).sum()

                # mse_loss = F.mse_loss(0.2*torch.log10(10 ** (5 * img_gen[0] - 4) + 10 ** (5 * img_gen[1] - 4))+0.8, self.original_imgs[0])

                mse_loss = ((0.2*torch.log10(10**(5*img_gen[0]-4) + 10**(5*img_gen[1]-4)) + 0.8 - self.original_imgs[0]) * self.mse_weight).pow(2).mean()

                p_loss = self.percept(.2*torch.log10(10 ** (5 * img_gen[0] - 4) + 10 ** (5 * img_gen[1] - 4))+0.8, self.original_imgs[0]).mean()
            elif self.task == 'denoise':
                diff = torch.abs(self.downsampler_1024_image(img_gen) - self.original_imgs) - self.dead_zone_linear_alpha
                dead_zone_linear_loss = torch.max(torch.zeros(diff.shape, device=diff.device), diff).mean()
                mse_loss = F.mse_loss(self.downsampler_1024_image(img_gen), self.original_imgs)
                p_loss = self.percept(self.downsampler_1024_256(img_gen), self.resized_imgs).mean()
            elif self.task == 'fast_compress':
                # TODO: check how to generalize for different sizes...
                perc = self.config['observed_percentage'] / 100
                m = int(perc * (self.config['image_size'][0] ** 2))
                indices = torch.tensor(np.random.choice(np.arange(1024 * 1024 * 3), m))
                filters = torch.ones((1024 *  1024 * 3), device=self.device).normal_().unsqueeze(0).to(self.device)
                real_obsv = partial_circulant_torch(self.original_imgs, filters, indices)
                gen_obsv = partial_circulant_torch(img_gen, filters, indices)
                mse_loss = F.mse_loss(real_obsv, gen_obsv)
                p_loss = torch.tensor(0.0)
                dead_zone_linear_loss = torch.tensor(0.0)

            # loss = self.dead_zone_linear * dead_zone_linear_loss + \
            #     self.config['pe'] * p_loss +  self.config['mse'] * mse_loss + \
            #     self.config['geocross'] * loss_geocross(self.latent_z[2 * start_layer:])

            mse_loss = 10 * torch.log10(mse_loss)
            loss = mse_loss + self.lpips_alpha * p_loss
            res.append(mse_loss.detach().cpu().item())

            if self.discriminator is not None:
                cls_loss = F.softplus(-self.discriminator(img_gen)).mean()
                loss = loss + self.cls_alpha * cls_loss

            if self.search_space in ["Z", "Z+"]:
                r_loss = self.latent_z.pow(2).mean()
                loss = loss + self.r_alpha * r_loss
            elif self.search_space in ["W+"]:
                r_loss = loss_geocross(latent_w[:, 2 * start_layer:], self.gen.style_dim)
                # r_loss = ((self.plrelu(latent_w)-self.mu) @ self.W).pow(2).mean()
                loss = loss + self.r_alpha * r_loss
            else:
                r_loss = torch.tensor(0., device='cuda')

            # if self.task == 'separate':
            #     loss = loss - 0.001 * (10**(5*img_gen[0]-4) + 10**(5*img_gen[1]-4)).pow(2).sum()

            # if self.config['cls']:
            #     downsampled = self.downsampler_1024_128(img_gen)
            #     cls_out = self.cls(downsampled)
            #     cls_loss = F.cross_entropy(cls_out, self.config['target'] * torch.ones(cls_out.shape[0], device=img_gen.device, dtype=torch.int64))
            #     loss += self.config['cls'] * cls_loss
            #     cls_prob = F.softmax(cls_out, dim=-1)[0, self.config['target']].item()
            # else:
            #     cls_prob = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.project:
                ps.step()

            if mse_loss < mse_min:
                mse_min = mse_loss
                self.best = img_gen.detach().cpu()
                no_improve = 0
            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f};"
                    f" r: {r_loss.detach().cpu().item():.4f} ;"
                    # f" cls_prob: {cls_loss:.4f} ;"
                    f"lr: {lr:.6f}"
                )
            )
            # if self.config['save_gif'] and i % self.config['save_every'] == 0:
            #     torchvision.utils.save_image(
            #         img_gen,
            #         f'gif_{start_layer}_{i}.png',
            #         nrow=int(img_gen.shape[0] ** 0.5),
            #         normalize=True)
            no_improve += 1
            if no_improve > patient:
                break
        # TODO: check what happens when we are in the last layer
        with torch.no_grad():
            latent_w = self.z_to_w(self.latent)
            self.gen.end_layer = self.gen.start_layer
            intermediate_out, _  = self.gen([latent_w],
                                             input_is_latent=True,
                                             noise=self.noises,
                                             layer_in=self.gen_outs[-1],
                                             skip=self.skip)
            intermediate_out = intermediate_out * self.scalar
            self.gen_outs.append(intermediate_out)
            self.gen.end_layer = self.end_layer

        return res

    def invert(self):
        res = []
        for i, steps in enumerate(self.steps.split(',')):
            begin_from = i + self.start_layer
            if begin_from > self.end_layer:
                raise Exception('Attempting to go after end layer...')
            res = self.invert_(begin_from, range(5 + 2 * begin_from), int(steps), res)
            # self.invert_(begin_from, range(3 + 2 * begin_from), int(steps))
        self.best[self.best<0] = 0
        return self.original_imgs, (self.latent, self.noises, self.gen_outs), self.best, res
