# ## SETUP

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



from tqdm import tqdm
from torch import optim
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from inspect import isfunction
from functools import partial
from palette.core.base_network import BaseNetwork
from palette.core.logger import LogTracker
from abc import abstractmethod
import tqdm



#from modules import UNet, EMA
from utils import *
from vq_model import Model, show_images
from palette.models.guided_diffusion_modules.unet import UNet
# -

# %%capture
from era5_precipitation_preprocessing import GFDL_P_Dataset_1_1, ERA5_P_Dataset
#from precipitation_preprocessing import Precipitation_Dataset, ISIMIP_P_Dataset_scale64

config = {"run_name": "DDPM_conditional",
          "epochs":        400,
          "batch_size":    8, 
          "lr":            1e-5, 
          "image_size":    64,             
          "num_classes":   10, 
          "device":        "cuda", 
          "num_workers":   8, 
}
#wandb.config.update({"image_size": 64})

wandb.init(project='climate-diffusion', entity='Michi',config=config, save_code=True)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# # Dataloaders

# +
train_set_p = ERA5_P_Dataset(stage='train')
train_set_p.get_mean_std()
all_data_p = train_set_p.data()

#n_val = int(len(all_data_p) * 0.1)
#n_train = len(all_data_p) - n_val

#train_dataset_p, val_dataset_p = random_split(all_data_p, [n_train, n_val], generator=torch.Generator().manual_seed(0))


dataloader_train_p = data.DataLoader(all_data_p, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                     num_workers=wandb.config.num_workers)

#dataloader_val_p = data.DataLoader(val_dataset_p, batch_size=136, shuffle=False, drop_last=True,
#                                   num_workers=wandb.config.num_workers)
next(iter(dataloader_train_p)).shape

# +
sample_era5 = next(iter(dataloader_train_p))

plt.hist(sample_era5[:100].flatten(),
                bins=100,
                histtype='step',
                log=True,
                density=True,
                linewidth=2)
plt.show()
# -

# days in 2020 + 2021: 366 + 365 = 731 
#
# so day 10957 - 731 days (2020,2021) = day 10226 which corresponds to date 31.12.2019
#
# day 10226 - 3287 days (2019.12-2011.01) = day 6939 = date 01.01.2011
#
# so validation period should be from day 6939-10226

6939-10226; len(dataloader_train_p)

bs_valid = 136

# +
era5_p = ERA5_P_Dataset(stage='valid')
era5_p.get_mean_std()
era5_p = era5_p.data()

dataloader_era5_val_p = data.DataLoader(era5_p, batch_size=bs_valid, shuffle=False, drop_last=True,
                                     num_workers=wandb.config.num_workers)

sample_era5 = next(iter(dataloader_era5_val_p))
print("era5:",sample_era5.shape)

# +
gfdl = GFDL_P_Dataset_1_1(stage='train')
gfdl.get_mean_std()
gfdl = gfdl.data()

dataloader_gfdl = data.DataLoader(gfdl, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=wandb.config.num_workers)

sample_gfdl = next(iter(dataloader_gfdl))

sample_gfdl = sample_gfdl.unsqueeze(dim=1)
sample_gfdl.shape


# -

# # Diffusion

class Diffusion_unconditional(BaseNetwork):
    def __init__(self, unet, beta_schedule, **kwargs):
        super(Diffusion_unconditional, self).__init__(**kwargs)
        #from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = unet #UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        #betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = self.beta_schedule
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm.tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            
            #y_t = torch.clamp(inverse_norm(y_t), 0, 128)     ###### do inverse trafo
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, noise=None):
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
            
            train_log, condition, original_img = self.train_step()
            print("epoch:", self.epoch)
            
            if self.epoch % 25 == 0:
                output_sampled, _ = diffusion_network.restoration(condition.to(config["device"]), sample_num=8)
                
                output_sampled_test = wandb.Image(output_sampled)
                wandb.log({"diffusion gen img": output_sampled_test})
                
                condition_wb = wandb.Image(condition)
                wandb.log({"condition img": condition_wb})
                
                original_img_wb = wandb.Image(original_img)
                wandb.log({"original img": original_img_wb})
                

                plt.imshow(output_sampled[1,0,:,:].cpu().detach().numpy())
                plt.title("diffusion generated sample")
                plt.show()
                
                plt.imshow(condition[1,0,:,:].cpu().detach().numpy())
                plt.title("condition sample")
                plt.show()
                
                print("original_img_wb",type(original_img))
                plt.imshow(original_img[1,0,:,:].cpu().detach().numpy())
                plt.title("original sample")
                plt.show()
                
                print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED")
                latitudinal_mean_three(original=original_img, generated=output_sampled, 
                                       label=condition.detach() , var="p")

                print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: UNET GENERATED")
                histograms_three(original=original_img.detach(), generated=output_sampled.detach(),
                                 label= condition.detach(),xlim_end=None, var="p")
                

                
                
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


def add_noise(y_0, noise_steps=500):

    noise = None
    b, *_ = y_0.shape
    t = torch.tensor([noise_steps], dtype=torch.long).to(device)

    gamma_t1 = extract(gammas, t-1, x_shape=(1, 1))
    sqrt_gamma_t2 = extract(gammas, t, x_shape=(1, 1))
    sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
    sample_gammas = sample_gammas.view(b, -1)

    noise = default(noise, lambda: torch.randn_like(y_0))
    y_noisy = q_sample(y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
    return y_noisy


class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, log_iter, model_path,
                  ema_scheduler=None,  **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.model_path = model_path
        self.log_iter = log_iter
        self.loss_fn = losses
        self.netG = networks

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
        for i, elements in enumerate(pbar):
            self.optG.zero_grad()
            self.gt_image  = elements
            self.gt_image = self.gt_image.unsqueeze(1).float().to(self.device)
            self.cond_image = add_noise(self.gt_image).float()
                    
            loss, y_noisy = self.netG(self.gt_image, self.cond_image)
            
            #y_noisy = wandb.Image(y_noisy)
            #wandb.log({"noisy_image": y_noisy})
            ## implement logging images cond_image + noise (need to modify self.netG to return the noised image)

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
            for i, elements in enumerate(pbar):
                self.gt_image = elements
                self.gt_image = self.gt_image.unsqueeze(1).float().to(self.device)
                self.cond_image = add_noise(self.gt_image).float()
                
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
unet_palette = UNet(image_size=64, in_channel=2,channel_mults=[1,2,4,8 ], inner_channel=64, 
     out_channel=1, res_blocks=2, num_head_channels=32, attn_res= [32,16,8], dropout= 0.2)


## Try cosine schedule  (was linear)

beta_schedule = make_beta_schedule("cosine", 2000, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3)

diffusion_network = Diffusion_unconditional(unet_palette,beta_schedule)
# -

device = wandb.config["device"]

# +
kwargs = {
    "phase": "train",
    "dataloader": dataloader_train_p, #dataloader_t, 
    "metrics": ["mae"],
    "resume_state" : True,
    "n_epochs" : 400,
    "batch_size" : config["batch_size"],
    "n_iter" : 10,
    "save_checkpoint_epoch" : 100, 
    #"save_checkpoint_epoch" : 25, 
}

palette_model = Palette(
    networks=diffusion_network,
    losses=mse_loss,
    sample_num=8,
    task="inpainting",
    optimizers={"lr": 5e-5, "weight_decay": 0},
    log_iter = 1000,
    model_path = "", #"models/DDPM_conditional/200_Diffusion_unconditional.pth",
    ema_scheduler=None,
    **kwargs
    )


# +
do_training = False


###dataloader_circ_1 = dataloader_slp, #dataloader_v850,    COMMENT IN !!!!

if do_training==True:
    palette_model_result = palette_model.train()
# -

# # Precipitation evaluation

# ## Try full gfdl valid dataset

# %%capture
"""
gfdl_test = GFDL_P_Dataset_1_1(stage='train')
gfdl_test.get_mean_std()
gfdl_test = gfdl_test.data()

dataloader_gfdl_test = data.DataLoader(gfdl_test, batch_size=500, shuffle=False, drop_last=True,num_workers=2)

gfdl_valid_all = next(iter(dataloader_gfdl_test))

gfdl_valid_all = gfdl_valid_all.unsqueeze(dim=1)
gfdl_valid_all.shape
"""

# +
#print("DM correction of full valid gfdl data")
#net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/400_Diffusion_unconditional.pth")
#bias_corr_gfdl, _  = net.restoration(gfdl_valid_all.to(device).float(), sample_num=gfdl_valid_all.shape[0])      
#plot_images_no_lab(bias_corr_gfdl[:5])

# +
#save_gfdl_dm_to_np = True
#print("save data for computing psd:",save_data_to_np)
#if save_gfdl_dm_to_np == True:
#    print("SAVING DATA TO DIRECTORY data/psd_model_data_precipitation")
#    np.save("data/psd_model_data/gfdl_raw_bs1000.npy", gfdl_inf.cpu().detach().numpy())
# -

# ## Run DM on GFDL

for b, el in enumerate(dataloader_gfdl):
            gfdl_inf = el
            gfdl_inf = gfdl_inf.unsqueeze(1)
            if b == 0:
                break

for batch, element in enumerate(dataloader_val_p):
            images_ = element
            original_images = images_.unsqueeze(1)
            if batch == 0:
                break

plot_images_no_lab(gfdl_inf[:5])

print("DM correction of gfdl")
net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/400_Diffusion_unconditional.pth")
bias_corr_gfdl, _  = net.restoration(gfdl_inf.to(device).float(), sample_num=gfdl_inf.shape[0])      
plot_images_no_lab(bias_corr_gfdl[:5])

print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
histograms_three(original=original_images.detach(), generated=bias_corr_gfdl.detach(), label= gfdl_inf.detach(),xlim_end=None, label_name=["era5","dm_gfdl","gfdl"],var="p")

print("BLUE: era5 VS ORANGE: DM bias corrected gfdl VS GREEN: GFDL")
latitudinal_mean_three(original=original_images[:120], generated=bias_corr_gfdl[:120], label=gfdl_inf[:120].detach(),label_name=["era5","dm_gfdl","gfdl"],var="p")
print("era5 lat mean - different from gfdl lat mean")

# # Save outputs (136,1,64,64)

save_data_to_np = False
print("save data for computing psd:",save_data_to_np)

if save_data_to_np == True:
    print("SAVING DATA TO DIRECTORY data/psd_model_data_precipitation")
    np.save("data/psd_model_data/gfdl_raw.npy", gfdl_inf.cpu().detach().numpy())
    np.save("data/psd_model_data/dm_gfdl_400.npy", bias_corr_gfdl.cpu().detach().numpy())
    np.save("data/psd_model_data/original_p_400.npy", original_images.numpy())
    np.save("data/psd_model_data/unet_p_400.npy", output.cpu().detach().numpy())
    np.save("data/psd_model_data/dm_p_400.npy", samples_t.cpu().numpy())

# # Condition on the noisy original image 

# +
t_int = 100

beta_schedule = make_beta_schedule("cosine", 2000, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3)

fw_dm_model = Diffusion_unconditional(unet_palette,beta_schedule)
fw_dm_model.to(device)
fw_dm_model.set_loss(mse_loss)
fw_dm_model.set_new_noise_schedule()



y_0 = original_images[:5].to(device)
noise = None
b, *_ = y_0.shape

#t = torch.randint(1, fw_dm_model.num_timesteps, (b,), device=y_0.device).long()
t = torch.tensor([t_int], dtype=torch.long).to(device)

print("timestep:",t)
gamma_t1 = extract(fw_dm_model.gammas, t-1, x_shape=(1, 1))
sqrt_gamma_t2 = extract(fw_dm_model.gammas, t, x_shape=(1, 1))
sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
sample_gammas = sample_gammas.view(b, -1)

noise = default(noise, lambda: torch.randn_like(y_0))
y_noisy = fw_dm_model.q_sample(y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
# -

plot_images_no_lab(y_noisy)

for batch, element in enumerate(zip(dataloader_era5_val_p , dataloader_gfdl)):
            original, gfdl = element     
            original = original.unsqueeze(1)
            gfdl = gfdl.unsqueeze(1)

            dm = palette_model.load_pretrain_diffusion("models/DDPM_conditional/400_Diffusion_unconditional.pth")
            
            noise = None
            b, *_ = original.shape
            
            t = torch.tensor([t_int], dtype=torch.long).to(device)

            print("timestep:",t)
            gamma_t1 = extract(fw_dm_model.gammas, t-1, x_shape=(1, 1))
            sqrt_gamma_t2 = extract(fw_dm_model.gammas, t, x_shape=(1, 1))
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
            sample_gammas = sample_gammas.view(b, -1)

            noise = default(noise, lambda: torch.randn_like(original)).to(device)
            y_noisy = fw_dm_model.q_sample(y_0=original.to(device), sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
            dm_samples, _  = dm.restoration(y_noisy.float(), sample_num=y_noisy.shape[0])
            
            if batch == 0:
                break

original.shape, y_noisy.shape, dm_samples.shape

plot_images_no_lab(y_noisy[:5])

plot_images_no_lab(original[:5])

plot_images_no_lab(dm_samples[:5])

plot_images_no_lab(gfdl[:5])

print("BLUE: ORIGINAL DATA VS ORANGE DIFFUSION VS GREEN:y_noisy")
latitudinal_mean_three(original=original, generated=dm_samples.detach(), label=y_noisy.detach(), label_name=["original","dm_samples","y_noisy"])

print("BLUE: ORIGINAL DATA VS ORANGE DIFFUSION VS GREEN:y_noisy")
histograms_three(original=original.detach(), generated=dm_samples.detach(), label= y_noisy.detach(),xlim_end=None, label_name=["original","DM","y_noisy"])

print("BLUE: ORIGINAL DATA VS ORANGE DIFFUSION VS GREEN:GFDL DATA")
latitudinal_mean_three(original=original, generated=dm_samples.detach(), label=gfdl.detach(), label_name=["original","dm_samples","GFDL"])
print("era5 lat mean - different from gfdl lat mean")

print("BLUE: ORIGINAL DATA VS ORANGE DIFFUSION VS GREEN:GFDL DATA")
histograms_three(original=original.detach(), generated=dm_samples.detach(), label= gfdl.detach(),xlim_end=None, label_name=["original","DM","GFDL"])

plt.imshow(torch.mean(gfdl,dim=0).squeeze())
plt.show()

# # Plot the mean images

# +
# Calculate the minimum and maximum values for scaling
min_value = torch.mean(original_images, axis=0).squeeze().cpu().min()
max_value = torch.mean(original_images, axis=0).squeeze().cpu().max()

# Create a figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image
im1 = axs[0].imshow(torch.mean(original_images, axis=0).squeeze().cpu(), vmin=min_value, vmax=max_value)
axs[0].set_title('Mean P over 100 original')

# Plot the second image with scaled values
im2 = axs[1].imshow(torch.mean(original_images, axis=0).squeeze() - torch.mean(dm_samples, axis=0).squeeze().cpu(),
                    vmin=min_value, vmax=max_value)
axs[1].set_title('Difference Mean P original - generated over 100 samples')

# Create a shared colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im1, cax=cax)

# Display the figure
plt.show()
# -

x = torch.mean(original_images, axis=0).squeeze() - torch.mean(dm_samples, axis=0).squeeze().cpu()
plt.hist(x)
plt.title("histogram over the mean difference between original vs diffusion generated data")
plt.show()

# +
# Calculate the minimum and maximum values for scaling
min_value = torch.mean(original_images, axis=0).squeeze().cpu().min()
max_value = torch.mean(original_images, axis=0).squeeze().cpu().max()

# Create a figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image
im1 = axs[0].imshow(torch.mean(original_images, axis=0).squeeze().cpu(), vmin=min_value, vmax=max_value)
axs[0].set_title('Mean T over 100 original')

# Plot the second image with scaled values
im2 = axs[1].imshow(torch.mean(dm_samples, axis=0).squeeze().cpu(),
                    vmin=min_value, vmax=max_value)
axs[1].set_title('Mean T over 100 generated')

# Create a shared colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im1, cax=cax)

# Display the figure
plt.show()
# -

print("original min,mean,max:",torch.min(original_images).item(), torch.mean(original_images).item(), torch.max(original_images).item())
print("generated min, mean, max:",torch.min(dm_samples).item(), torch.mean(dm_samples).item(), torch.max(dm_samples).item())




