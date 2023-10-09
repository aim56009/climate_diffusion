# # Imports

# %%capture
# !pip install xarray

import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils import data
from src.base_network import BaseNetwork
from src.raw_diffusion_functions import make_beta_schedule, Diffusion_unconditional, Palette
from src.unet import UNet
from src.utils import *
from src.dataloaders import GFDL_P_Dataset_1_1, ERA5_P_Dataset
from src.dataloaders import Antialiasing, dwd_rv_rainrate_transform
from src.dataloaders import BC_GFDL_Dataset_256
from src.dataloaders import U850_Dataset, V850_Dataset, SLP_Dataset
from src.condition_unet_loading import UNet_condition
from src.dataloaders import BC_UNET_VALID_Dataset_64
from src.dataloaders import UNET_VALID_Dataset_64

# # Dataloader

bs_valid = 10

# +
gfdl_dataset = GFDL_P_Dataset_1_1(stage='train')
gfdl_dataset.get_mean_std()
gfdl_dataset_ = gfdl_dataset.data()

dataloader_gfdl = data.DataLoader(gfdl_dataset_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)

sample_gfdl = next(iter(dataloader_gfdl))

sample_gfdl = sample_gfdl.unsqueeze(dim=1)
sample_gfdl.shape

# +
era5_p = ERA5_P_Dataset(stage='valid')
era5_p.get_mean_std()
era5_p_ = era5_p.data()

dataloader_era5_val_p = data.DataLoader(era5_p_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)

sample_era5 = next(iter(dataloader_era5_val_p))
print("era5:",sample_era5.shape)
# -

# # Load Model

# +
unet_palette = UNet(image_size=64, in_channel=2,channel_mults=[1,2,4,8 ], inner_channel=64, 
     out_channel=1, res_blocks=2, num_head_channels=32, attn_res= [32,16,8], dropout= 0.2)


beta_schedule = make_beta_schedule("cosine", 2000, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3)

diffusion_network = Diffusion_unconditional(unet_palette,beta_schedule)


# -

def mse_loss(output, target):
    return F.mse_loss(output, target)


# +
kwargs = {
    "phase": "train",
    "dataloader": "", #dataloader_t, 
    "metrics": ["mae"],
    "resume_state" : True,
    "n_epochs" : 400,
    "batch_size" : 8,
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
    model_path = "", 
    dataloader_circ_1 = "", 
    dataloader_circ_2 = "",
    dataloader_circ_3 = "",
    unet_condition_encoder = "",
    ema_scheduler=None,
    **kwargs
    )
device = "cuda"
# -

# # Do inference

do_single_sample_inf = False

if do_single_sample_inf==True:
    for b, el in enumerate(dataloader_gfdl):
            gfdl_inf = el
            gfdl_inf = gfdl_inf.unsqueeze(1)
            if b == 0:
                break
    plot_images_no_lab(gfdl_inf[:5])
    print(gfdl_inf.shape)

if do_single_sample_inf==True:
    print("DM correction of gfdl")
    net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/400_Diffusion_unconditional_best_p.pth")
    bias_corr_gfdl, _  = net.restoration(gfdl_inf.to(device).float(), sample_num=gfdl_inf.shape[0])      
    plot_images_no_lab(bias_corr_gfdl[:5])

if do_single_sample_inf==True:
    bias_corr_gfdl.shape, original_images.shape, gfdl_inf.shape

if do_single_sample_inf==True:
    for batch, element in enumerate(dataloader_era5_val_p):
                original_images = element.unsqueeze(1)

if do_single_sample_inf==True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three(original=original_images.detach()
                     , generated=bias_corr_gfdl.detach()
                     , label= gfdl_inf.detach(),xlim_end=None, label_name=["era5","dm_gfdl","gfdl"],var="p")

# # RUN DM on whole GFDL dataset

do_save_bc_gfdl_dataset = False

# +
net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/400_Diffusion_unconditional_best_p.pth")

print("MODEL USED TO CREATE BC GFDL: DDPM_conditional/400_Diffusion_unconditional.pth")
# -

if do_save_bc_gfdl_dataset==True:
    output_tensors = []

    for b, el in enumerate(dataloader_gfdl):
        gfdl_inf = el.unsqueeze(1).to(device).float()

        bias_corr_gfdl_output, _ = net.restoration(gfdl_inf, sample_num=gfdl_inf.shape[0])
        output_tensors.append(bias_corr_gfdl_output)
        #if b == 1:
        #    break

    bias_corr_gfdl_dataset = torch.cat(output_tensors, dim=0) 
    
    print(bias_corr_gfdl_dataset.shape)

if do_save_bc_gfdl_dataset==True:
    torch.save(bias_corr_gfdl_dataset, 'data/bias_corr_gfdl_dataset.pth')

# # Load bc gfdl data

loaded_dataset = torch.load('data/bias_corr_gfdl_dataset.pth')
print("loaded data - shape:",loaded_dataset.shape)

# +
bc_gfdl_dataset = BC_GFDL_Dataset_256("valid")
bc_gfdl_dataset_ = bc_gfdl_dataset.data()

dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)

# +
plot_bc_gfdl_sample=False

if plot_bc_gfdl_sample==True:
    bc_gfld_sample = next(iter(dataloader_bc_gfdl))
    print("batch size:",bc_gfld_sample.shape)
    plt.imshow(bc_gfld_sample[0,:,:].cpu())
    plt.show()
# -

for b, el in enumerate(dataloader_gfdl):
            gfdl_inf = el
            gfdl_inf = gfdl_inf.unsqueeze(1)
            if b == 0:
                break

if plot_bc_gfdl_sample==True:
    bc_gfdl_ori_unit = bc_gfdl_dataset.inverse_dwd_trafo(bc_gfld_sample)
    van_gfdl_ori_unit = gfdl_dataset.inverse_dwd_trafo(gfdl_inf)
    ear5_ori_unit = era5_p.inverse_dwd_trafo(sample_era5)

    histograms_three(bc_gfdl_ori_unit, van_gfdl_ori_unit, ear5_ori_unit, 
                     label_name=['bc_gfdl_ori_unith', 'original gfdl_dataset', 'era5'])
    print("Just for 1 batch ")

if plot_bc_gfdl_sample==True:
    histograms_three(bc_gfld_sample, gfdl_inf, sample_era5, 
                     label_name=['bc gfdl 1 batch', 'original gfdl 1 batch', 'original sample_era5'])
    print("Just for 1 batch ")

# # Save UNET validation dataset

bs_valid_unet = 1

# +
create_unet_dataset = False      ## DONT SET TRUE ---> should be no mistake here 


if create_unet_dataset==True:
    output_tensors_unet = []

    compare_original_slp = SLP_Dataset(stage='valid')
    dataloader_original_val_snap_slp = data.DataLoader(compare_original_slp.data(), batch_size=bs_valid_unet, shuffle=False, drop_last=True,num_workers=2)

    compare_original_u = U850_Dataset(stage='valid')
    dataloader_original_val_snap_u850 = data.DataLoader(compare_original_u.data(), batch_size=bs_valid_unet, shuffle=False, drop_last=True,num_workers=2)

    compare_original_v = V850_Dataset(stage='valid')
    dataloader_original_val_snap_v850 = data.DataLoader(compare_original_v.data(), batch_size=bs_valid_unet, shuffle=False, drop_last=True,num_workers=2)


    fn_tonumpy = lambda x:x.to('cpu').detach().numpy()
    loaded_model = UNet_condition(nch=3,nker=64,out_chan=1,norm="bnorm")#.to(device)
    state_dict = torch.load("models/unet_before/ckpt90.pt")
    loaded_model.load_state_dict(state_dict)
    net = loaded_model


    for batch, element in enumerate(zip(dataloader_original_val_snap_v850, dataloader_original_val_snap_u850, dataloader_original_val_snap_slp)):
            cond_1, cond_2, cond_3 = element
            cond_1 = cond_1.unsqueeze(1)
            cond_2 = cond_2.unsqueeze(1)
            cond_3 = cond_3.unsqueeze(1)
            image = torch.cat((cond_1, cond_2, cond_3), dim=1).float()#.to(device)            
            unet_partial = net(image)   
            output_tensors_unet.append(unet_partial)
            if batch % 100==0:
                print(batch)
            #if batch == 1:
            #    break
    UNET_era5_lr_64_dataset = torch.cat(output_tensors_unet, dim=0)
            
    print("UNET ERA5 LR 64 dataset shape:", UNET_era5_lr_64_dataset.shape)
# -

if create_unet_dataset==True:  ### comment in when really wanting to save
    #torch.save(UNET_era5_lr_64_dataset.to("cuda"), 'data/UNET_era5_lr_64_dataset.pth')
    print("saving")

loaded_dataset = torch.load('data/UNET_era5_lr_64_dataset.pth')
print("loaded data - shape:",loaded_dataset.shape)

# # RUN DM on UNET validation dataset

# +
unet_val_dataset = UNET_VALID_Dataset_64("valid")

dataloader_unet_val = data.DataLoader(unet_val_dataset.data(), batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)
unet_val_sample = next(iter(dataloader_unet_val))
unet_val_sample.shape
# -

do_save_bc_unet_dataset = False

# +
net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/400_Diffusion_unconditional_best_p.pth")

print("MODEL USED TO CREATE BC GFDL: DDPM_conditional/400_Diffusion_unconditional.pth")
# -

if do_save_bc_unet_dataset==True:
    output_tensors_unet = []

    for b, el in enumerate(dataloader_unet_val):
        unet_val = el.unsqueeze(1).to(device).float()

        bias_corr_unet_output, _ = net.restoration(unet_val, sample_num=unet_val.shape[0])
        output_tensors_unet.append(bias_corr_unet_output)
        #if b == 1:
        #    break

    bias_corr_unet_val_dataset = torch.cat(output_tensors_unet, dim=0) 
    print(bias_corr_unet_val_dataset.shape)


if do_save_bc_unet_dataset==True:
    ### comment in when really wanting to save
    #torch.save(bias_corr_unet_val_dataset, 'data/bias_corr_unet_val_dataset.pth')
    print("saving")

loaded_dataset = torch.load('data/bias_corr_unet_val_dataset.pth')
print("loaded data - shape:",loaded_dataset.shape)

# +
bc_unet_val_dataset = BC_UNET_VALID_Dataset_64("valid")

dataloader_bc_unet_val = data.DataLoader(bc_unet_val_dataset.data(), batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)
next(iter(dataloader_bc_unet_val)).shape

# +
plot_bc_unet_sample=True

if plot_bc_unet_sample==True:
    bc_unet_sample = next(iter(dataloader_bc_unet_val))
    print("batch size:",bc_unet_sample.shape)
    plt.imshow(bc_unet_sample[0,:,:].cpu())
    plt.show()
# -

if plot_bc_unet_sample==True:
    histograms_three( unet_val_dataset.inverse_dwd_trafo(unet_val_sample)
                     ,bc_unet_val_dataset.inverse_dwd_trafo(bc_unet_sample)
                     ,era5_p.inverse_dwd_trafo(sample_era5) 
                     ,label_name=['UNET valid', 'BC UNET valid', 'ERA5 valid'])
    print("Just for 1 batch ")

if plot_bc_unet_sample==True:
    histograms_three( unet_val_sample
                     ,bc_unet_sample
                     ,sample_era5 
                     ,label_name=['UNET valid', 'BC UNET valid', 'ERA5 valid'])
    print("Just for 1 batch ")

latitudinal_mean_three_np( unet_val_sample.unsqueeze(1).numpy()
                          ,bc_unet_sample.unsqueeze(1).numpy()
                          ,sample_era5.unsqueeze(1).numpy()
                          ,label_name=['UNET valid', 'BC UNET valid', 'ERA5 valid'])




