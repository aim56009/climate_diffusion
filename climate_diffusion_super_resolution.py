# # SETUP

do_training = False

load_pretrained_model = True

# %%capture
# !pip install xarray
# !pip install wandb
# !pip install netcdf4
# !pip install collections
# !pip install scikit-image

# +
import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import xarray as xr
import wandb
import IPython.display as display
import logging
import torch.nn.functional as F
import collections
import copy
import torchvision.transforms as transforms


from PIL import Image
from tqdm import tqdm
from torch import optim
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from inspect import isfunction
from functools import partial
from abc import abstractmethod
from skimage.metrics import structural_similarity as ssim

import tqdm
from src.utils import *
from src.condition_unet_loading import UNet_condition

from src.unet import UNet
from src.base_network import BaseNetwork
# -

# #%%capture
from src.dataloaders import GFDL_P_Dataset_1_1, ERA5_P_Dataset, ERA5_P_0_25_to_1_Dataset, ERA5_P_0_25_Dataset
from src.dataloaders import Antialiasing, dwd_rv_rainrate_transform, ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256
from src.dataloaders import ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_gauss_blur
from src.dataloaders import ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_blur_pix
from src.dataloaders import BC_GFDL_Dataset_256
from src.dataloaders import GFDL_P_Dataset_to_256

# +
#from apply_bias_correction_to_gfdl import BC_GFDL_Dataset
# -

config = {"run_name": "SR_diffusion",     # folder name  
          "epochs":        400,
          "batch_size":    4, 
          "lr":            1e-5, 
          "image_size":    256,             
          "num_classes":   10, 
          "device":        "cuda", 
          "num_workers":   8, 
}
#wandb.config.update({"image_size": 64})

wandb.init(project='climate-diffusion', entity='Michi',config=config, save_code=True)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# # Load the conditional UNet (pretrained)

train_era5_yu_64 = False
if train_era5_yu_64 == True:
    train_set_p = ERA5_P_Dataset(stage='train')
    train_set_p.get_mean_std()
    all_data_p = train_set_p.data()

    dataloader_train_p = data.DataLoader(all_data_p, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    next(iter(dataloader_train_p)).shape

train_era5_1d_64 = False
if train_era5_1d_64 == True:
    era5_p_trafo_1 = ERA5_P_0_25_to_1_Dataset(stage='train')
    era5_p_trafo_1.get_mean_std()
    era5_p_trafo_1_ = era5_p_trafo_1.data()

    dataloader_era5_train_trafo_1 = data.DataLoader(era5_p_trafo_1_, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5_trafo_1 = next(iter(dataloader_era5_train_trafo_1))
    print(sample_era5_trafo_1.shape)

# +
era5_p025_tr = ERA5_P_0_25_Dataset(stage='train')
era5_p025_tr.get_mean_std()
era5_p025_tr_ = era5_p025_tr.data()

dataloader_era5_train_p025 = data.DataLoader(era5_p025_tr_, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                     num_workers=wandb.config.num_workers)

sample_era5_025_tr = next(iter(dataloader_era5_train_p025))
print(sample_era5_025_tr.shape)
# -

era5_train_1d_256_upsample = True
if era5_train_1d_256_upsample == True:
    era5_p_1d_256 = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(stage='train')
    era5_p_1d_256.get_mean_std()
    era5_p_1d_256_ = era5_p_1d_256.data()

    dataloader_era5_train_1d_256 = data.DataLoader(era5_p_1d_256_, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5_1d_256p = next(iter(dataloader_era5_train_1d_256))
    print(sample_era5_1d_256p.shape)

# ##### train era5 1degree with 256 pixel, blured

train_just_blur = False
if train_just_blur == True:
    era5_p_1d_256_blur = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_gauss_blur(stage='train')
    era5_p_1d_256_blur.get_mean_std()
    era5_p_1d_256_blur = era5_p_1d_256_blur.data()

    dataloader_era5_train_1d_256_blur = data.DataLoader(era5_p_1d_256_blur, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5_1d_256p_blur = next(iter(dataloader_era5_train_1d_256_blur))
    print(sample_era5_1d_256p_blur.shape)

# ##### train era5 1degree with 256 pixel, blur & pixelate

# +
train_blur_and_pix = False

if train_just_blur == True:
    era5_p_1d_256_pix = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_blur_pix(stage='train')
    era5_p_1d_256_pix.get_mean_std()
    era5_p_1d_256_pix = era5_p_1d_256_pix.data()

    dataloader_era5_train_1d_256_blur_pix = data.DataLoader(era5_p_1d_256_pix, batch_size=wandb.config.batch_size,
                                                            shuffle=False, drop_last=True,
                                                            num_workers=wandb.config.num_workers)

    sample_era5_1d_256p_pix = next(iter(dataloader_era5_train_1d_256_blur_pix))
    print(sample_era5_1d_256p_pix.shape)
# -

# ## validation

bs_valid = 8

# ### validate bias corrected gfdl

# +
#loaded_dataset = torch.load('data/bias_corr_gfdl_dataset.pth') # data/ !!
#print("loaded data - shape:",loaded_dataset.shape)

bc_gfdl_dataset = BC_GFDL_Dataset_256("valid")
bc_gfdl_dataset.get_mean_std()
bc_gfdl_dataset_ = bc_gfdl_dataset.data()


dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=bs_valid, 
                                     shuffle=False, drop_last=True, 
                                     num_workers=wandb.config.num_workers)

bc_gfld_sample = next(iter(dataloader_bc_gfdl))
print("batch size:",bc_gfld_sample.shape)
# -

# ##### era5 1degree with 256 pixel , blured

train_just_blur = False
if train_just_blur == True:
    era5_p_1d_256_v_blur = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_gauss_blur(stage='valid')
    era5_p_1d_256_v_blur.get_mean_std()
    era5_p_1d_256_v_blur = era5_p_1d_256_v_blur.data()

    dataloader_era5_val_1d_256_blur = data.DataLoader(era5_p_1d_256_v_blur, batch_size=bs_valid, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5_1d_256p_val_blur = next(iter(dataloader_era5_val_1d_256_blur))
    print(sample_era5_1d_256p_val_blur.shape)

# ##### era5 1degree with 256 pixel , blure & pixelated

train_blur_pixel = False
if train_blur_pixel == True:
    era5_p_1d_256_v_pix = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_blur_pix(stage='valid')
    era5_p_1d_256_v_pix.get_mean_std()
    era5_p_1d_256_v_pix = era5_p_1d_256_v_pix.data()

    dataloader_era5_val_1d_256_blur_pix = data.DataLoader(era5_p_1d_256_v_pix, batch_size=bs_valid, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5_1d_256p_val_pix = next(iter(dataloader_era5_val_1d_256_blur_pix))
    print(sample_era5_1d_256p_val_pix.shape)

# ##### normal data (no blur)

# +
era5_p_1d_256_v = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(stage='valid')
era5_p_1d_256_v.get_mean_std()
era5_p_1d_256_v_ = era5_p_1d_256_v.data()

dataloader_era5_val_1d_256 = data.DataLoader(era5_p_1d_256_v_, batch_size=bs_valid, shuffle=False, drop_last=True,
                                     num_workers=wandb.config.num_workers)

sample_era5_1d_256p_val = next(iter(dataloader_era5_val_1d_256))
print(sample_era5_1d_256p_val.shape)

# +
era5_p025 = ERA5_P_0_25_Dataset(stage='valid')
era5_p025.get_mean_std()
era5_p025_ = era5_p025.data()

dataloader_era5_val_p025 = data.DataLoader(era5_p025_, batch_size=bs_valid, shuffle=False, drop_last=True,
                                     num_workers=wandb.config.num_workers)

sample_era5_025 = next(iter(dataloader_era5_val_p025))
print(sample_era5_025.shape)

# +
#era5_p_trafo_1 = ERA5_P_0_25_to_1_Dataset(stage='valid')
#era5_p_trafo_1.get_mean_std()
#era5_p_trafo_1_ = era5_p_trafo_1.data()

#dataloader_era5_val_trafo_1 = data.DataLoader(era5_p_trafo_1_, batch_size=bs_valid, shuffle=False, drop_last=True,
#                                     num_workers=wandb.config.num_workers)

#sample_era5_trafo_1 = next(iter(dataloader_era5_val_trafo_1))
#print(sample_era5_trafo_1.shape)
# -

valid_era5_yu_64 = False
if valid_era5_yu_64 == True:
    era5_p = ERA5_P_Dataset(stage='valid')
    era5_p.get_mean_std()
    era5_p_ = era5_p.data()

    dataloader_era5_val_p = data.DataLoader(era5_p_, batch_size=bs_valid, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5 = next(iter(dataloader_era5_val_p))
    print("era5:",sample_era5.shape)

# +
load_vanilla_gfdl = True

if load_vanilla_gfdl == True:
    gfdl = GFDL_P_Dataset_1_1(stage='train')  # stage train should be equal to valid
    gfdl.get_mean_std()
    gfdl_ = gfdl.data()

    dataloader_gfdl = data.DataLoader(gfdl_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=wandb.config.num_workers)

    sample_gfdl = next(iter(dataloader_gfdl))

    sample_gfdl = sample_gfdl#.unsqueeze(dim=1)
    sample_gfdl.shape
# -

plt.imshow(sample_gfdl[0,:,:].cpu())
plt.show()

plt.imshow(bc_gfld_sample[0,:,:].cpu())
plt.show()


# # Diffusion

class Diffusion_SR(BaseNetwork):
    def __init__(self, unet, beta_schedule, **kwargs):
        super(Diffusion_SR, self).__init__(**kwargs)
        #from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = unet #UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        """ set_new_noise_schedule initializes gamma, secduler values 
            in paper: line under Eq.(4) and line under Eq.(5)
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        #betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = self.beta_schedule
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        """ in paper: line under Eq.(4)"""
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        """ in paper line under Eq.(5)"""
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        """ in paper: Eq.(8) - noise: will be unet([y_cond,y_t],gamma_t)"""
        """output: y_0_hat"""
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        """ in paper: line under Eq.(5) mu
            q_posterior return: mu, sigm**2
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        """ fct: sample noise -> get y_0_hat -> get mu_theta,sigma_theta**2
            return:mu_theta, sigm_theta**2  (NN parametrization of p_theta(y_t-1|y_t, x);x=y_cond)
        """
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        """denoising unet network -in paper: NN in Algo 2 line 4 """
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))
    
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)
        """take y_0_hat from Eq.8 and put it into mean, sig calc. under Eq.(5) -> mu_theta, sig_theta"""
        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance


    def q_sample(self, y_0, sample_gammas, noise=None):
        """in paper: Eq.(6) (forward diffusion process)"""
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        """ in paper: Eq.(9) (reverse diffusion process) Algo 4 line 2"""
        """output: y_0 by doing t steps of rev diff on y_{t-1} """
        """Method: compute Eq.(8) using unet parametrz of noise -> plug into mu,sig under Eq.(5) -> 
        parametrize to get p_theta(y_t-1|y_t, x) with mu_theta, sig_theta to get 
        -> Eq.(5) p_theta(y_t-1|y_t, x)=N(y_t-1|mu_theta,sig_theta**2I) 
        p_theta(y_t-1|y_t, x) is gaussian => sample by: y_t-1 = mu_theta + sigma_theta * noise """
        
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        """ second part: inverse of log_var = ln(sigma**2 ) -> sig = e**(0.5*log_var)"""
        """sample from reverse diffusion process p_theta(y_t-1|y_t, x) (guassian): N(mu,sig) = x + noise * sig """
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, sample_num=8):
        """in paper: Algorithm 2"""
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm.tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            """in paper: Algorithm 2 line 4 !"""
            y_t = self.p_sample(y_t, t, y_cond=y_cond) #"""for 1st iteration: y_t contains gaussian noise,(for all elements in batch)"""
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            
            #y_t = torch.clamp(inverse_norm(y_t), 0, 128)     ###### do inverse trafo
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, noise=None):
        """in paper: Algorithm 1"""
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        loss = self.loss_fn(noise, noise_hat)
        return loss, y_noisy


# +
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


# -

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# # Training

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class BaseModel():
    def __init__(self, phase,  dataloader, metrics, n_epochs=10, batch_size = 8, n_iter=10,
                  save_checkpoint_epoch=10,resume_state=False, **kwargs):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.phase = phase
        self.device = config['device']
        
        self.n_epochs = n_epochs
        self.n_iter = n_iter
        self.resume_state = resume_state
        self.save_checkpoint_epoch = save_checkpoint_epoch

        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []

        ''' process record '''
        self.batch_size = batch_size
        self.epoch = 0
        self.iter = 0 

        self.phase_loader = dataloader
        self.metrics = metrics

        ''' logger to log file, which only work on GPU 0. writer to tensorboard and result file '''
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}

    def train(self):
        while self.epoch <= self.n_epochs: #and self.iter <= self.n_iter:
            self.epoch += 1
            
            train_log, condition_after_unet, original_img = self.train_step()
            print("epoch:", self.epoch)
            
            if self.epoch % 25 == 0:
                output_sampled, _ = diffusion_network.restoration(condition_after_unet.to(config["device"]), sample_num=8)
                
                output_sampled_test = wandb.Image(output_sampled)
                wandb.log({"diffusion gen img": output_sampled_test})
                
                condition_after_unet_wb = wandb.Image(condition_after_unet)
                wandb.log({"condition img": condition_after_unet_wb})
                
                original_img_wb = wandb.Image(original_img)
                wandb.log({"original img": original_img_wb})
                

                plt.imshow(output_sampled[1,0,:,:].cpu().detach().numpy())
                plt.title("diffusion generated sample")
                plt.show()
                
                plt.imshow(condition_after_unet[1,0,:,:].cpu().detach().numpy())
                plt.title("condition sample")
                plt.show()
                
                print("original_img_wb",type(original_img))
                plt.imshow(original_img[1,0,:,:].cpu().detach().numpy())
                plt.title("original sample")
                plt.show()
                
                print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED")
                latitudinal_mean_three(original=original_img, generated=output_sampled, 
                                       label=condition_after_unet.detach() , var="p")

                print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: UNET GENERATED")
                histograms_three(original=original_img.detach(), generated=output_sampled.detach(),
                                 label= condition_after_unet.detach(),xlim_end=None, var="p")
                
                #lat_mean_wb = wandb.Image(latitudinal_mean_three(original=original_img, generated=output_sampled, 
                #                                                 label=condition_after_unet.detach(), var="p"))
                #wandb.log({"latitudinal mean": lat_mean_wb})
                
                
                
            if self.epoch % self.save_checkpoint_epoch == 0:
                print('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()


    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')


    def test_step(self):
        pass
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        print('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        print(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        save_filename = '{}_{}.pth'.format(self.epoch, network_label)
        save_path = os.path.join(os.path.join("models", config['run_name'], save_filename))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, model_path, strict=True):        
        if not os.path.exists(model_path):
            print('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return

        print('Loading pretrained model from [{:s}] ...'.format(model_path))
        network.load_state_dict(torch.load(model_path), strict=strict)
        network.to(self.device)

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """

        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(self.epoch)
        save_path = os.path.join(os.path.join("models", config['run_name'], save_filename))
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase!='train' or self.resume_state is None:
            return
        print('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_path = "{}.state".format(self.resume_state)
        
        if not os.path.exists(state_path):
            print('Training state in [{:s}] is not existed, Skip it'.format(state_path))
            return

        print('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path).to(self.device) 
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers {} != {}'.format(len(resume_schedulers), len(self.schedulers))
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    def load_everything(self):
        pass 
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')


class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, log_iter, model_path, dataloader_circ_1,
                 ema_scheduler=None,  **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.model_path = model_path
        self.log_iter = log_iter
        self.loss_fn = losses
        self.netG = networks
        self.dataloader_circ_1 = dataloader_circ_1

        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG.to(self.device)
        if self.ema_scheduler is not None:
            self.netG_EMA = self.netG.to(self.device) 
        self.load_networks(self.model_path)

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers)
        self.optimizers.append(self.optG)
        self.resume_training() 

        self.netG.set_loss(self.loss_fn)
        self.netG.set_new_noise_schedule()

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = data.get('cond_image').to(self.device)
        self.gt_image = data.get('gt_image').to(self.device)
    
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }

        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        pbar = tqdm.tqdm(self.phase_loader)
        for i, elements in enumerate(zip(pbar,self.dataloader_circ_1)):
            self.optG.zero_grad()
            self.gt_image, self.cond_image_1  = elements
            self.gt_image = self.gt_image.unsqueeze(1).float().to(self.device)
            #self.cond_image_1 = self.cond_image_1.unsqueeze(1).float().to(self.device)
            self.cond_image = self.cond_image_1.unsqueeze(1).float().to(self.device)
            
            #print("self.cond_image_1",self.cond_image_1.shape)
            #self.cond_image = F.interpolate(self.cond_image_1, scale_factor=4, mode='nearest') # increase size by factor 4 , 64->256
            #print("self.cond_image_1",self.cond_image.shape)
            
                    
            loss, y_noisy = self.netG(self.gt_image, self.cond_image)
            

            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            
            if i % self.log_iter == 0:
                print(loss.item())
                wandb.log({"loss": loss.item()})
                
                #output_sampled = diffusion_network.restoration(test_condition.to(config["device"]), sample_num=8)
     
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
            
        return self.netG, self.cond_image, self.gt_image

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(self.phase_loader)
            for i, elements in enumerate(zip(pbar,self.dataloader_circ_1)):
                self.gt_image, self.cond_image_1  = elements
                self.gt_image = self.gt_image.unsqueeze(1).float().to(self.device)
                #self.cond_image_1 = self.cond_image_1.unsqueeze(1).float().to(self.device)
                self.cond_image = self.cond_image_1.unsqueeze(1).float().to(self.device)
                
                #self.cond_image = self.cond_image_1.float()
                #self.cond_image = F.interpolate(self.cond_image_1, scale_factor=4, mode='nearest') # increase size by factor 4 , 64->256

                self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                if i == 1:
                    break
        

        ''' print logged informations to the screen and tensorboard ''' 
        return self.output, self.visuals

    def load_networks(self, model_path):
        """ save pretrained model and training state, which only do on GPU 0. """
        netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, model_path=model_path, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)
          
        
    def load_pretrain_diffusion(self, model_path):
        self.netG.load_state_dict(torch.load(model_path), strict=False)
        self.netG.to(self.device)
        
        if self.ema_scheduler is not None:
            self.netG_EMA.load_state_dict(torch.load(model_path), strict=False)
            self.netG_EMA.to(self.device)
            return self.netG_EMA
        return self.netG
        
    
    
    def save_everything(self):
        """ load pretrained model and training state. """
        netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

CustomResult = collections.namedtuple('CustomResult', 'name result')


def mse_loss(output, target):
    return F.mse_loss(output, target)


# +
unet_palette = UNet(image_size=256, in_channel=2,channel_mults=[1,2,4,8 ], inner_channel=64, 
     out_channel=1, res_blocks=2, num_head_channels=32, attn_res= [32,16,8], dropout= 0.2)


## Try cosine schedule  (was linear)

beta_schedule = make_beta_schedule("cosine", 2000, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3)

diffusion_network = Diffusion_SR(unet_palette,beta_schedule)
# -

device = wandb.config["device"]

# +
kwargs = {
    "phase": "train",
    "dataloader": dataloader_era5_train_p025, #dataloader_t, 
    "metrics": ["mae"],
    "resume_state" : True,
    "n_epochs" : 400, #400,
    "batch_size" : config["batch_size"],
    "n_iter" : 10,
    "save_checkpoint_epoch" : 50, 
    #"save_checkpoint_epoch" : 25, 
}

palette_model = Palette(
    networks=diffusion_network,
    losses=mse_loss,
    sample_num=8,
    task="inpainting",
    optimizers={"lr": 5e-5, "weight_decay": 0},
    log_iter = 1000,
    model_path = "",  #"models/DDPM_conditional/191_Diffusion_unconditional.pth"
    dataloader_circ_1 = dataloader_era5_train_1d_256, ## should NOT be dataloader_bc_gfdl 
    ema_scheduler=None,
    **kwargs
    )
# -


diffusion_network.__class__.__name__ ### chaning saving name -> diffusion_network = SR_diffusion(unet_palette,beta_schedule)

model_condition_name = "dataloader_era5_train_1d_256"

diffusion_network.__class__.__name__ = model_condition_name
diffusion_network.__class__.__name__

# +
do_training = False


if do_training==True:
    palette_model_result = palette_model.train()

# +
## currently test 1 go DM: lr gfdl -> HR BC gfdl -> "models/SR_diffusion/350_Diffusion_SR.pth"
# -

# # Precipitation evaluation

# ## Run DM on era5 (era5 lr (pix), era5 hr, era5 DM hr)

# +
run_dm = False


data_blur = False

# +
if data_blur == False:
    era5_lr = next(iter(dataloader_era5_val_1d_256)).unsqueeze(1)    ## for normal unblured data
if data_blur == True:
    era5_lr = next(iter(dataloader_era5_val_1d_256_blur_pix)).unsqueeze(1)

era5_lr.shape
# -

era5_hr = next(iter(dataloader_era5_val_p025)).unsqueeze(1)
era5_hr.shape

era5_hr.shape, era5_lr.shape

inv_trafo = True

if run_dm == True:
    print("DM correction of lr era5, data_blur:",data_blur)
    if data_blur == True:
        net = palette_model.load_pretrain_diffusion("models/SR_diffusion/291_Diffusion_sr.pth")
    if data_blur == False:
        net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/400_Diffusion_sr.pth")
    
    dm_hr_era5, _  = net.restoration(era5_lr.to(device).float(), sample_num=era5_lr.shape[0])      
    #plot_images_no_lab(dm_hr_era5[:5])

if run_dm == True:
    print("era5 LR")
    plot_images_no_lab(era5_lr[:5])
    print("era5 HR")
    plot_images_no_lab(era5_hr[:5])
    print("era5 DM HR")
    plot_images_no_lab(dm_hr_era5[:5])

# +
#plot_images_no_lab(dm_hr_era5[:5])
# -

if run_dm == True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three_np(original=era5_p025.inverse_dwd_trafo(era5_hr.numpy()).flatten()
         ,generated=era5_p025.inverse_dwd_trafo(dm_hr_era5.cpu().numpy()).flatten()
         ,label= era5_p_1d_256_v.inverse_dwd_trafo(era5_lr.numpy()).flatten()
         ,xlim_end=None, label_name=["ear5 hr"," dm_hr_era5","era5 lr"],var="p")

if run_dm == True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    latitudinal_mean_three_np(original=era5_p025.inverse_dwd_trafo(era5_hr.numpy())
         ,generated=era5_p025.inverse_dwd_trafo(dm_hr_era5.cpu().numpy())
         ,label= era5_p_1d_256_v.inverse_dwd_trafo(era5_lr.numpy())
         ,label_name=["ear5 hr"," dm_hr_era5","era5 lr"],var="p")

if run_dm == True and inv_trafo == False:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three(original=era5_hr.detach().flatten(), generated=dm_hr_era5.detach().flatten(), label= era5_lr.detach().flatten()
                     ,xlim_end=None, label_name=["ear5 hr","dm era5 hr","era5 lr"],var="p")

if run_dm == True and inv_trafo == False:
    latitudinal_mean_three(original=era5_hr.detach(), generated=dm_hr_era5.detach(), 
                       label=era5_lr.detach()
                 ,label_name=["ear5 hr","dm era5 hr","era5 lr"],var="p")

if run_dm == True and inv_trafo == False:
    plt.imshow(era5_hr[0,0,:,:].cpu()-dm_hr_era5[0,0,:,:].cpu())
    plt.title("1 sample: difference between 0.25d era5 and DM downscaled (bc+upsample) 0.25d era5")
    plt.colorbar()
    plt.show()
    print("mean difference over 8 samples", torch.mean(era5_hr[:,0,:,:].cpu()-dm_hr_era5[:,0,:,:].cpu()))

# ## compute the SSIM era5 and dm era5 hr

if run_dm == True:
    ssim_scores_era5_hr_lr = np.zeros(era5_hr.shape[0])

    for i in range(8):
        era5_hr_np = era5_hr[i].cpu().numpy()  
        dm_hr_era5_np = dm_hr_era5[i].cpu().numpy()  
        ssim_scores_era5_hr_lr[i] = ssim(era5_hr_np[0], dm_hr_era5_np[0], data_range=era5_hr_np.max() - era5_hr_np.min())

    # ssim_scores now contains the SSIM scores for each pair as a NumPy array
    print("ssim_scores:",ssim_scores_era5_hr_lr)
    print("batch avg ssim score:",np.mean(ssim_scores_era5_hr_lr))

if run_dm == True:
    mse = np.zeros(era5_hr.shape[0])

    for i in range(8):
        era5_hr_np = era5_hr[i].cpu().numpy()  
        dm_hr_era5_np = dm_hr_era5[i].cpu().numpy() 
        squared_diff = (era5_hr_np - dm_hr_era5_np) ** 2  # Calculate squared differences
        mse[i] = np.mean(squared_diff) 

    print("mse:",mse)
    print("avg mse:",np.mean(mse))

# ### SSIM betwenn era5 lr - hr

if run_dm == True:
    ssim_scores_era5_hr_lr = np.zeros(era5_hr.shape[0])

    for i in range(8):
        era5_hr_np = era5_hr[i].cpu().numpy()  
        era5_lr_np = era5_lr[i].cpu().numpy()  
        ssim_scores_era5_hr_lr[i] = ssim(era5_hr_np[0], era5_lr_np[0], data_range=era5_hr_np.max() - era5_lr_np.min())

    # ssim_scores now contains the SSIM scores for each pair as a NumPy array
    print("ssim_scores:",ssim_scores_era5_hr_lr)
    print("batch avg ssim score:",np.mean(ssim_scores_era5_hr_lr))

# ### Save outputs (8,1,256,256)

save_data_to_np = False
print("save data for computing psd:",save_data_to_np)

if save_data_to_np == True:
    print("SAVING DATA TO DIRECTORY data/psd_model_data_precipitation")
    np.save("data/psd_model_data/era5_hr.npy", era5_hr.cpu().detach().numpy())
    np.save("data/psd_model_data/dm_hr_era5.npy", dm_hr_era5.cpu().detach().numpy())
    np.save("data/psd_model_data/era5_lr.npy", era5_lr.numpy())

# # evaluate on bias corrected GFDL (era5 hr , gfdl lr , gfdl DM hr)

# +
gfdl_test = GFDL_P_Dataset_to_256(stage='valid')
gfdl_test_ = gfdl_test.data()

dataloader_gfdl_test = data.DataLoader(gfdl_test_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)

gfdl_lr_vanilla_256 = next(iter(dataloader_gfdl_test))
gfdl_lr_vanilla_256 = gfdl_lr_vanilla_256.unsqueeze(dim=1)
print(gfdl_lr_vanilla_256.shape)
# -

eval_bc_gfdl = False

if eval_bc_gfdl == True:
    loaded_dataset = torch.load('data/bias_corr_gfdl_dataset.pth') # data/ !!
    print("loaded data - shape:",loaded_dataset.shape)

    bc_gfdl_dataset = BC_GFDL_Dataset_256(loaded_dataset)
    bc_gfdl_dataset.get_mean_std()
    bc_gfdl_dataset_ = bc_gfdl_dataset.data()


    dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=bs_valid, 
                                         shuffle=False, drop_last=True, 
                                         num_workers=wandb.config.num_workers)

    bc_gfld_sample = next(iter(dataloader_bc_gfdl))    
    print("DM super resolution of bias corrected gfdl")
    
    bc_gfld_sample = bc_gfld_sample.unsqueeze(dim=1)
    print(bc_gfld_sample.shape)
    
    net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")
    #net = palette_model.load_pretrain_diffusion("models/SR_diffusion/350_Diffusion_SR.pth")
    dm_hr_bc_gfdl, _  = net.restoration(bc_gfld_sample.to(device).float(), sample_num=bc_gfld_sample.shape[0])      
    plot_images_no_lab(dm_hr_bc_gfdl[:5])

plot_images_no_lab(gfdl_lr_vanilla_256[:5])
print("lr gfdl data")

original_units = True
#eval_bc_gfdl = True

if eval_bc_gfdl and original_units == True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three_np(original=era5_p025.inverse_dwd_trafo(era5_hr.numpy()).flatten()
                        ,generated=era5_p025.inverse_dwd_trafo(dm_hr_bc_gfdl.cpu().numpy()).flatten() 
                        ,label= gfdl_test.inverse_dwd_trafo(gfdl_lr_vanilla_256.numpy()).flatten()
                        ,xlim_end=None, label_name=["era5 hr","gfdl dm hr","gfdl lr"],var="p")

if eval_vanilla_gfdl == True and original_units == False:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three(original=era5_hr.detach().flatten(), 
                     generated=dm_hr_bc_gfdl.detach().flatten(), 
                     label= gfdl_lr_vanilla_256.detach().flatten()
                     ,xlim_end=None, label_name=["era5 hr","gfdl dm hr","gfdl lr"],var="p")

# ### compute the SSIM era5- bc gfdl hr

if eval_bc_gfdl == True:
    ssim_scores_era5_gfdl_hr = np.zeros(era5_hr.shape[0])

    for i in range(8):
        era5_hr_np = era5_hr[i].cpu().numpy()  
        dm_hr_era5_np = dm_hr_bc_gfdl[i].cpu().numpy()  
        ssim_scores_era5_gfdl_hr[i] = ssim(era5_hr_np[0], dm_hr_era5_np[0], data_range=era5_hr_np.max() - dm_hr_era5_np.min())

    # ssim_scores now contains the SSIM scores for each pair as a NumPy array
    print("ssim_scores:",ssim_scores_era5_gfdl_hr)
    print("batch avg ssim score:",np.mean(ssim_scores_era5_gfdl_hr))

if eval_bc_gfdl == True:
    mse = np.zeros(era5_hr.shape[0])

    for i in range(8):
        era5_hr_np = era5_hr[i].cpu().numpy()  
        dm_hr_gfdl_np = dm_hr_bc_gfdl[i].cpu().numpy() 
        squared_diff = (era5_hr_np - dm_hr_gfdl_np) ** 2  # Calculate squared differences
        mse[i] = np.mean(squared_diff) 

    print("mse:",mse)
    print("avg mse:",np.mean(mse))

if eval_bc_gfdl == True:
    print("dm_hr_bc_gfdl", dm_hr_bc_gfdl.shape)
    plot_images_no_lab(dm_hr_bc_gfdl[:5])
    print("gfdl_lr_vanilla_256", gfdl_lr_vanilla_256.shape)
    plot_images_no_lab(gfdl_lr_vanilla_256[:5])
    print("era5_lr", era5_lr.shape)
    plot_images_no_lab(era5_lr[:5])

# ### save HR bias corrected gfdl data to numpy

# +
save_bc_gfdl_data_to_numpy = False
print("save data for computing psd:",save_bc_gfdl_data_to_numpy)

if save_bc_gfdl_data_to_numpy == True:
    print("SAVING DATA TO DIRECTORY data/psd_model_data_precipitation")
    np.save("data/psd_model_data/dm_hr_bc_gfdl.npy", dm_hr_bc_gfdl.cpu().detach().numpy())
# -

# # evaluate on vanilla gfdl (era5 hr , gfdl lr , gfdl DM hr)

eval_vanilla_gfdl = True

if eval_vanilla_gfdl == True:
    #from dataloaders import GFDL_P_Dataset_to_256
    #gfdl_test = GFDL_P_Dataset_to_256(stage='valid')
    #gfdl_test_ = gfdl_test.data()
    #dataloader_gfdl_test = data.DataLoader(gfdl_test_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)
    #gfdl_lr_vanilla_256 = next(iter(dataloader_gfdl_test))
    #gfdl_lr_vanilla_256 = gfdl_lr_vanilla_256.unsqueeze(dim=1)
    #print(gfdl_lr_vanilla_256.shape)
    
    print("DM super resolution of vanilla gfdl")
    net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")
    #net = palette_model.load_pretrain_diffusion("models/SR_diffusion/350_Diffusion_SR.pth")
    dm_hr_vanilla_gfdl, _  = net.restoration(gfdl_lr_vanilla_256.to(device).float(), sample_num=gfdl_lr_vanilla_256.shape[0])      
    plot_images_no_lab(dm_hr_vanilla_gfdl[:5])

# +
#plot_images_no_lab(dm_hr_vanilla_gfdl[:5])
# -

original_units = False

### try dm.inverse_dwd .. ->  ,generated=era5_p025.inverse_dwd_trafo(dm_hr_vanilla_gfdl.cpu().numpy()).flatten() 
if eval_vanilla_gfdl and original_units == True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three_np(original=era5_p025.inverse_dwd_trafo(era5_hr.numpy()).flatten()
                        ,generated=era5_p025.inverse_dwd_trafo(dm_hr_vanilla_gfdl.cpu().numpy()).flatten() 
                        ,label= gfdl_test.inverse_dwd_trafo(gfdl_lr_vanilla_256.numpy()).flatten()
                        ,xlim_end=None, label_name=["era5 hr","gfdl dm hr","gfdl lr"],var="p")

if eval_vanilla_gfdl == True and original_units == False:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three(original=era5_hr.detach().flatten(), generated=dm_hr_vanilla_gfdl.detach().flatten(), 
                     label= gfdl_lr_vanilla_256.detach().flatten()
                     ,xlim_end=None, label_name=["era5 hr","gfdl dm hr","gfdl lr"],var="p")

# +
#if eval_vanilla_gfdl == True:
#    latitudinal_mean_three(original=era5_hr.detach(), generated=dm_hr_vanilla_gfdl.detach(), 
#                       label=gfdl_lr_vanilla_256.detach()
#                 ,label_name=["ear5 hr","dm era5 hr","era5 lr"],var="p")
# -

# ### compute the average mean SSIM era5-gfdl hr

# +
ssim_scores_era5_gfdl_hr = np.zeros(era5_hr.shape[0])

mean_era5_hr_np = np.mean(era5_hr.cpu().numpy()[:,0,:,:],axis=0)
mean_dm_hr_gfdl_np = np.mean(dm_hr_vanilla_gfdl.cpu().numpy()[:,0,:,:],axis=0)
ssim_scores_era5_gfdl_hr = ssim(mean_era5_hr_np, mean_dm_hr_gfdl_np, data_range=mean_era5_hr_np.max() - mean_dm_hr_gfdl_np.min())

# ssim_scores now contains the SSIM scores for each pair as a NumPy array
print("ssim_scores:",ssim_scores_era5_gfdl_hr)
print("batch avg ssim score:",np.mean(ssim_scores_era5_gfdl_hr))
# -

# ## comparison LR gfdl & era5 (no DM)

# potential problem: LR gfdl is way more blury than LR era5 (64x64) 
# -> look at psd: prbl. gfdl diverges fast from era5, so goal: make psd's more similar for longer by bluring LR era5 for training -> inference will be better

# +
print("dm_hr_vanilla_gfdl", dm_hr_vanilla_gfdl.shape)
plot_images_no_lab(dm_hr_vanilla_gfdl[:5])
print("gfdl_lr_vanilla_256", gfdl_lr_vanilla_256.shape)
plot_images_no_lab(gfdl_lr_vanilla_256[:5])
print("era5_lr", era5_lr.shape)
plot_images_no_lab(era5_lr[:5])

pixel_size = 4  # Adjust this value to control the pixelation level
# Resize the image tensor to a smaller size
resized_image_era5_lr = F.interpolate(era5_lr, size=(256 // pixel_size, 256 // pixel_size), mode='nearest')
# Upscale the resized image back to the original size
pixelated_image_era5_lr = F.interpolate(resized_image_era5_lr, size=(256, 256), mode='nearest')
print("pixelated era5_lr", pixelated_image_era5_lr.shape)
plot_images_no_lab(pixelated_image_era5_lr[:5])
# -

# ## Test bluring vs pixelation of era5

# +
image = era5_lr

"""Apply Gaussian blur with a specified sigma"""
sig = 2.0
blurred_image = transforms.GaussianBlur(kernel_size=23, sigma=(sig, sig))(image)

"""Pixelate era5_lr"""
pixel_size = 3  # Adjust this value to control the pixelation level
# Resize the image tensor to a smaller size
resized_image = F.interpolate(image, size=(256 // pixel_size, 256 // pixel_size), mode='nearest')
# Upscale the resized image back to the original size
pixelated_image = F.interpolate(resized_image, size=(256, 256), mode='nearest')

"""Pixelate blured era5_lr"""
resized_blured_image = F.interpolate(blurred_image, size=(256 // pixel_size, 256 // pixel_size), mode='nearest')
pixelated_blured_image = F.interpolate(resized_blured_image, size=(256, 256), mode='nearest')


plt.figure(figsize=(10, 10))
grid = plt.GridSpec(2, 3)
# Subplot 1: Original 'era5_lr' image
plt.subplot(grid[0, 0])
plt.imshow(era5_lr[0, 0, :, :])
plt.title("Original lr era5")
plt.axis('off')
# Subplot 2: 'blurred_image'
plt.subplot(grid[0, 1])
plt.imshow(blurred_image[0, 0, :, :])
plt.title("blurred lr era5")
plt.axis('off')
# Subplot 3: 'pixelated_image'
plt.subplot(grid[0, 2])
plt.imshow(pixelated_image[0, 0, :, :])
plt.title("pixelated era5 lr")
plt.axis('off')
# Subplot 4: 'pixelated_blurred_image'
plt.subplot(grid[1, 0])
plt.imshow(pixelated_blured_image[0, 0, :, :])
plt.title("pixelated and blurred era5 lr")
plt.axis('off')
# Subplot 5: 'gfdl_lr_vanilla_256'
plt.subplot(grid[1, 1])
plt.imshow(gfdl_lr_vanilla_256[0, 0, :, :])
plt.title("lr gfdl")
plt.axis('off')
plt.tight_layout()
plt.show()
# -

histograms_three(original=era5_lr.detach().flatten(), generated=pixelated_blured_image.detach().flatten(), 
                     label= gfdl_lr_vanilla_256.detach().flatten()
                     ,xlim_end=None, label_name=["era5 lr","blured + pix era5 lr","gfdl lr"],var="p")

eval_vanilla_gfdl = True

if eval_vanilla_gfdl == True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three(original=era5_hr.detach().flatten(), generated=gfdl_lr_vanilla_256.detach().flatten(), 
                     label= era5_lr.detach().flatten()
                     ,xlim_end=None, label_name=["ear5 hr","gfdl lr vanilla","era5 lr"],var="p")

if eval_vanilla_gfdl and original_units == True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three_np(original=era5_p025.inverse_dwd_trafo(era5_hr.numpy()).flatten()
                        ,generated=era5_p025.inverse_dwd_trafo(gfdl_lr_vanilla_256.cpu().numpy()).flatten() 
                        ,label= gfdl_test.inverse_dwd_trafo(era5_lr.numpy()).flatten()
                        ,xlim_end=None, label_name=["era5 hr","gfdl lr 256","era5 lr"],var="p")

# +
#if eval_vanilla_gfdl == True:
#    latitudinal_mean_three(original=era5_hr.detach(), generated=gfdl_lr_vanilla_256.detach(), 
#                       label=era5_lr.detach()
#>                 ,label_name=["ear5 hr","gfdl lr vanilla","era5 lr"],var="p")
# -

# ### compute the SSIM era5-gfdl lr

# +
### MAYBE DO AN AVERAGE OVER MANY LR ERA5 LR GFDL AND THEN CALCULATE THE SSIM -> MEAN SSIM
ssim_scores_era5_gfdl_lr = np.zeros(era5_lr.shape[0])

for i in range(8):
    era5_lr_np = era5_lr[i].cpu().numpy()  
    dm_lr_gfdl_np = gfdl_lr_vanilla_256[i].cpu().numpy()  
    ssim_scores_era5_gfdl_lr[i] = ssim(era5_lr_np[0], dm_lr_gfdl_np[0], data_range=era5_lr_np.max() - dm_lr_gfdl_np.min())

# ssim_scores now contains the SSIM scores for each pair as a NumPy array
print("ssim_scores:",ssim_scores_era5_gfdl_lr)
print("batch avg ssim score:",np.mean(ssim_scores_era5_gfdl_lr))
# -

# ## save LR gfdl data to numpy

# +
save_data_to_numpy = False
print("save data for computing psd:",save_data_to_numpy)

if save_data_to_numpy == True:
    print("SAVING DATA TO DIRECTORY data/psd_model_data_precipitation")
    np.save("data/psd_model_data/gfdl_lr_vanilla_256.npy", gfdl_lr_vanilla_256.cpu().detach().numpy())
    np.save("data/psd_model_data/dm_hr_vanilla_gfdl.npy", dm_hr_vanilla_gfdl.cpu().detach().numpy())
# -


# ## Try bigger BS for gfdl valid dataset

# +
try_valid_custom_bs = False
new_bs = 50

if try_valid_custom_bs == True:
    era5_hr_bs = ERA5_P_0_25_Dataset(stage='valid')
    era5_hr_bs.get_mean_std()
    era5_hr_bs = era5_hr_bs.data()

    dataloader_era5_val_hr_bs = data.DataLoader(era5_hr_bs, batch_size=new_bs, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5_hr_bs = next(iter(dataloader_era5_val_hr_bs))
    sample_era5_hr_bs = sample_era5_hr_bs.unsqueeze(dim=1)
    print(sample_era5_hr_bs.shape)
    
    
    ear5_lr_bs = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(stage='valid')
    ear5_lr_bs.get_mean_std()
    ear5_lr_bs = ear5_lr_bs.data()

    dataloader_era5_val_lr_bs = data.DataLoader(ear5_lr_bs, batch_size=new_bs, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    sample_era5_lr_bs = next(iter(dataloader_era5_val_lr_bs))
    sample_era5_lr_bs = sample_era5_lr_bs.unsqueeze(dim=1)
    print(sample_era5_lr_bs.shape)
    
    print("DM correction of more valid LR era5 data")
    #net_sr = palette_model.load_pretrain_diffusion("models/DDPM_conditional/191_Diffusion_unconditional.pth")
    net_sr = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")
    sr_era5_dm, _  = net_sr.restoration(sample_era5_1d_256p_val.to(device).float(), 
                                                   sample_num=sample_era5_1d_256p_val.shape[0])      
    plot_images_no_lab(sr_era5_dm[:5])
    
 
    print("more vanilla gfdl data")
    gfdl_test_bs = GFDL_P_Dataset_to_256(stage='valid')
    gfdl_test_bs_ = gfdl_test_bs.data()

    dataloader_gfdl_test_bs = data.DataLoader(gfdl_test_bs_, batch_size=new_bs, shuffle=False, drop_last=True,num_workers=2)

    gfdl_lr_vanilla_256_bs = next(iter(dataloader_gfdl_test_bs))

    gfdl_lr_vanilla_256_bs = gfdl_lr_vanilla_256_bs.unsqueeze(dim=1)
    print(gfdl_lr_vanilla_256_bs.shape)
    
    
    
    
    print("more bc gfdl data")
    loaded_dataset_bs = torch.load('bias_corr_gfdl_dataset.pth') # data/ !!

    bc_gfdl_dataset_bs = BC_GFDL_Dataset_256(loaded_dataset_bs)
    bc_gfdl_dataset_bs.get_mean_std()
    bc_gfdl_dataset_bs_ = bc_gfdl_dataset_bs.data()

    dataloader_bc_gfdl_bs = data.DataLoader(bc_gfdl_dataset_bs_, batch_size=new_bs, 
                                         shuffle=False, drop_last=True, 
                                         num_workers=wandb.config.num_workers)

    bc_gfld_sample_bs = next(iter(dataloader_bc_gfdl_bs))    
    print("DM super resolution of bias corrected gfdl")
    bc_gfld_sample_bs = bc_gfld_sample_bs.unsqueeze(dim=1)
    
    net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")
    dm_hr_bc_gfdl_bigger_bs, _  = net.restoration(bc_gfld_sample_bs.to(device).float(), sample_num=bc_gfld_sample_bs.shape[0])      
    print(dm_hr_bc_gfdl_bigger_bs.shape)
    plot_images_no_lab(dm_hr_bc_gfdl_bigger_bs[:5])

# +
print("more bc gfdl data")
loaded_dataset_bs = torch.load('data/bias_corr_gfdl_dataset.pth') # data/ !!

bc_gfdl_dataset_bs = BC_GFDL_Dataset_256(loaded_dataset_bs)
bc_gfdl_dataset_bs.get_mean_std()
bc_gfdl_dataset_bs_ = bc_gfdl_dataset_bs.data()

dataloader_bc_gfdl_bs = data.DataLoader(bc_gfdl_dataset_bs_, batch_size=new_bs, 
                                     shuffle=False, drop_last=True, 
                                     num_workers=wandb.config.num_workers)

bc_gfld_sample_bs = next(iter(dataloader_bc_gfdl_bs))    
print("DM super resolution of bias corrected gfdl")
bc_gfld_sample_bs = bc_gfld_sample_bs.unsqueeze(dim=1)

net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")
dm_hr_bc_gfdl_bigger_bs, _  = net.restoration(bc_gfld_sample_bs.to(device).float(), sample_num=bc_gfld_sample_bs.shape[0])      
print(dm_hr_bc_gfdl_bigger_bs.shape)
plot_images_no_lab(dm_hr_bc_gfdl_bigger_bs[:5])
# -

np.save("data/psd_model_data/dm_hr_bc_gfdl_bigger_bs.npy", dm_hr_bc_gfdl_bigger_bs.cpu().detach().numpy())

save_gfdl_dm_to_np = False
print("save data for computing psd:",save_gfdl_dm_to_np)
if save_gfdl_dm_to_np == True:
    print("SAVING DATA TO DIRECTORY data/psd_model_data_precipitation")
    np.save("data/psd_model_data/dm_era5_bigger_bs.npy", sr_era5_dm.cpu().detach().numpy())
    np.save("data/psd_model_data/era5_lr_bigger_bs.npy", sample_era5_lr_bs)
    np.save("data/psd_model_data/era5_hr_bigger_bs.npy", sample_era5_hr_bs)
    np.save("data/psd_model_data/gfdl_lr_vanilla_256_bs.npy", gfdl_lr_vanilla_256_bs)
    np.save("data/psd_model_data/dm_hr_bc_gfdl_bigger_bs.npy", dm_hr_bc_gfdl_bigger_bs.cpu().detach().nu mpy())

# # Add Max's constraint

from src.constraints import Constraint


class Diffusion_unconditional_constraint(BaseNetwork):
    def __init__(self, unet, beta_schedule, **kwargs):
        super(Diffusion_unconditional, self).__init__(**kwargs)
        #from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = unet #UNet(**unet)
        self.beta_schedule = beta_schedule
        self.constraint = Constraint()
        
    def setup_constraint(self, u_0, t_0=0.0):
        """Set ups the constraint, if it is configured 

        Args: 
            config (Config): stores hyperparameters and file paths
            u_0: Initial conditions for the constraint: tensor 1x1xHxW. Used to compute the value of the constraint at u_0 and t_0
            t_0: Initial conditions for the constraint: tensor 1x1xHxW
        """
        self.constraint = Constraint(u_0, t_0)
        self.use_constraint = True
        

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        """ set_new_noise_schedule initializes gamma, secduler values 
            in paper: line under Eq.(4) and line under Eq.(5)
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        #betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = self.beta_schedule
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        """ in paper: line under Eq.(4)"""
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        """ in paper line under Eq.(5)"""
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        """ in paper: Eq.(8) - noise: will be unet([y_cond,y_t],gamma_t)"""
        """output: y_0_hat"""
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        """ in paper: line under Eq.(5) mu
            q_posterior return: mu, sigm**2
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        """ 
        fct: sample noise -> get y_0_hat -> get mu_theta,sigma_theta**2
        return:mu_theta, sigm_theta**2  (NN parametrization of p_theta(y_t-1|y_t, x);x=y_cond)
        """
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        """denoising unet network -in paper: NN in Algo 2 line 4 """
        
        if (self.constraint is not None) and self.constraint.use_at_inference: 
                y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, noise=(self.denoise_fn(torch.cat([y_cond, y_t], dim=1)- self.constraint(x, time_step)), noise_level))
        
        else:
            y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))
        ## check if concat with condition is consistent in code: cat(cond,y_t) not cat(y_t,cond)
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)
        """take y_0_hat from Eq.8 and put it into mean, sig calc. under Eq.(5) -> mu_theta, sig_theta"""
        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        """in paper: Eq.(6) (forward diffusion process)"""
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        """ in paper: Eq.(9) (reverse diffusion process) Algo 4 line 2"""
        """output: y_0 by doing t steps of rev diff on y_{t-1} """
        """Method: compute Eq.(8) using unet parametrz of noise -> plug into mu,sig under Eq.(5) -> 
        parametrize to get p_theta(y_t-1|y_t, x) with mu_theta, sig_theta to get 
        -> Eq.(5) p_theta(y_t-1|y_t, x)=N(y_t-1|mu_theta,sig_theta**2I) 
        p_theta(y_t-1|y_t, x) is gaussian => sample by: y_t-1 = mu_theta + sigma_theta * noise """
        
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        """ second part: inverse of log_var = ln(sigma**2 ) -> sig = e**(0.5*log_var)"""
        """sample from reverse diffusion process p_theta(y_t-1|y_t, x) (guassian): N(mu,sig) = x + noise * sig """
        return model_mean + noise * (0.5 * model_log_variance).exp()
        
    
    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, sample_num=8):
        """in paper: Algorithm 2"""
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm.tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            """in paper: Algorithm 2 line 4 !"""
            y_t = self.p_sample(y_t, t, y_cond=y_cond) #"""for 1st iteration: y_t contains gaussian noise,(for all elements in batch)"""
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            
            #y_t = torch.clamp(inverse_norm(y_t), 0, 128)     ###### do inverse trafo
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, noise=None):
        """in paper: Algorithm 1"""
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        loss = self.loss_fn(noise, noise_hat)
        return loss, y_noisy

# +
# mu_after = mu_bevor + noise
# mu_bevor = predicted -> mu_bevor - constr = mu_after - noise
# -



