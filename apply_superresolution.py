# Notebook to create a dataset fully with bias corrected high resolution GFDL data !
#
# 1.  train BC-DM:     cond=UNET/noisy lr version of era5 (1d), target=lr era5 (1d)
#     inference BC-DM: cond=lr gfdl (1d), output=lr gfdl data with statistics of era5 (1d)
#
# 2.  create a Dataset with the BC lr GFDL data AND upsampled to 256x256 pixels (still 1degree)
#     apply_bias_correction_to_gfdl.py is run. 
#     Runs the BC-DM on full lr gfdl dataset.
#    
# 3.  train SR-DM:     cond=lr era5 (1d), target=hr era5 (0.25d)
#                      inference=lr bc gfdl (1d), output= hr bc gfdl (0.25d)
#
# 4. Run apply_SR_to_bc_gfdl.py to run SR-DM on the hr bc gfdl dataset. 
#
# HR BC GFDL = SR-DM(BC-DM(LR GFDL))  => 1.run apply_bias_correction_to_gfdl -> 2. run apply_SR_to_bc_gfdl

# ## Imports

# %%capture
# #!pip install xarray
# !pip install skimage
import skimage

# +
from torch.utils import data
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from src.utils import *
from src.dataloaders import Antialiasing, dwd_rv_rainrate_transform
from src.base_network import BaseNetwork
from src.raw_diffusion_functions import make_beta_schedule, Diffusion_unconditional, Palette
from src.unet import UNet
from src.dataloaders import ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256
from src.dataloaders import BC_GFDL_Dataset_256, ERA5_P_0_25_Dataset, SR_BC_GFDL_Dataset_256, SR_ERA5_HR_Dataset_256
from src.psd_utils import SpatialSpectralDensity_diff_res, SpatialSpectralDensity_4_diff_res
# -

# ## Dataloaders

bs_valid = 10

loaded_dataset = torch.load('data/bias_corr_gfdl_dataset.pth')
print("loaded data - shape:",loaded_dataset.shape)

# +
bc_gfdl_dataset = BC_GFDL_Dataset_256(loaded_dataset)
bc_gfdl_dataset_ = bc_gfdl_dataset.data()

dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)

# +
era5_p025 = ERA5_P_0_25_Dataset(stage='valid')
era5_p025_ = era5_p025.data()

dataloader_era5_val_p025 = data.DataLoader(era5_p025_, batch_size=bs_valid, shuffle=False, drop_last=True,num_workers=2)

era5_hr_256 = next(iter(dataloader_era5_val_p025)).unsqueeze(1).numpy()
print("ERA5 HR 256 shape:",era5_hr_256.shape)
# -

# ## Load Model

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

# ## Inference

do_single_sample_inf = True

if do_single_sample_inf==True:
    for b, el in enumerate(dataloader_bc_gfdl):
            gfdl_inf = el
            gfdl_inf = gfdl_inf.unsqueeze(1)
            if b == 0:
                break
    plot_images_no_lab(gfdl_inf[:5])
    print(gfdl_inf.shape)

if do_single_sample_inf==True:
    print("DM correction of gfdl")
    net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")
    bias_corr_gfdl, _  = net.restoration(gfdl_inf.to(device).float(), sample_num=gfdl_inf.shape[0])      
    plot_images_no_lab(bias_corr_gfdl[:5])

if do_single_sample_inf==True:
    bias_corr_gfdl.shape, gfdl_inf.shape

if do_single_sample_inf==True:
    print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: GFDL")
    histograms_three(original=bias_corr_gfdl.detach()
                     , generated=gfdl_inf.detach()
                     , label= gfdl_inf.detach(),xlim_end=None, label_name=["bias_corr_gfdl","dm_gfdl","dm gfdl"],var="p")

# ## RUN on whole bc gfdl dataset

do_save_bc_gfdl_dataset = False

# +
net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")

print("MODEL USED TO CREATE BC GFDL: DDPM_conditional/291_Diffusion_sr.pth")
# -

dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=bs_valid, 
                                     shuffle=False, 
                                     drop_last=False,
                                     num_workers=2)

if do_save_bc_gfdl_dataset==True:
    output_tensors = []

    for b, el in enumerate(dataloader_bc_gfdl):
        gfdl_inf = el.unsqueeze(1).to(device).float()

        bias_corr_gfdl_output, _ = net.restoration(gfdl_inf, sample_num=gfdl_inf.shape[0])
        output_tensors.append(bias_corr_gfdl_output)
        if b%10==0:
            print(b)
        #if b == 1:
        #    break

    bias_corr_hr_gfdl_dataset = torch.cat(output_tensors, dim=0) 
    
    print(bias_corr_hr_gfdl_dataset.shape)

if do_save_bc_gfdl_dataset==True:
    print("uncomment for saving")
    #torch.save(bias_corr_hr_gfdl_dataset, 'data/HR_BC_GFDL_dataset.pth')

# ## Load HR BC GFDL Data

hr_bc_gfdl_dataset = torch.load('data/HR_BC_GFDL_dataset.pth')
print("loaded data - shape:",hr_bc_gfdl_dataset.shape)

# # Do validation

batch_size_val = 20

# +
hr_bc_gfdl_dataset = SR_BC_GFDL_Dataset_256(hr_bc_gfdl_dataset)
hr_bc_gfdl_dataset_ = hr_bc_gfdl_dataset.data()

dataloader_hr_bc_gfdl = data.DataLoader(hr_bc_gfdl_dataset_, batch_size=batch_size_val, shuffle=False, drop_last=True,num_workers=2)
hr_bc_gfld_sample = next(iter(dataloader_hr_bc_gfdl))
hr_bc_gfld_sample.shape
# -

plt.imshow(hr_bc_gfld_sample[0,:,:].cpu())
plt.show()

# +
dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=batch_size_val, shuffle=False, drop_last=True,num_workers=2)

bc_gfdl_sample = next(iter(dataloader_bc_gfdl)).unsqueeze(1)
bc_gfdl_sample.shape

# +
era5_p025 = ERA5_P_0_25_Dataset(stage='valid')
era5_p025_ = era5_p025.data()

dataloader_era5_val_p025 = data.DataLoader(era5_p025_, batch_size=batch_size_val, shuffle=False, drop_last=True,num_workers=2)

era5_hr_256 = next(iter(dataloader_era5_val_p025)).unsqueeze(1).numpy()
print("ERA5 HR 256 shape:",era5_hr_256.shape)
# -

histograms_three( hr_bc_gfld_sample
                 ,bc_gfdl_sample
                 ,torch.from_numpy(era5_hr_256), 
                 label_name=['HR-BC GFDL', 'BC GFDL', 'ERA5 HR'])

histograms_three( hr_bc_gfdl_dataset.inverse_dwd_trafo(hr_bc_gfld_sample)
                 ,bc_gfdl_dataset.inverse_dwd_trafo(bc_gfdl_sample)
                 ,torch.from_numpy(era5_p025.inverse_dwd_trafo(era5_hr_256))
                 ,label_name=['HR-BC GFDL', 'BC GFDL', 'ERA5 HR'])

resized_bc_gfdl_lr = []
for i in range(bc_gfdl_sample.shape[0]):
            original_image = bc_gfdl_sample[i, :, :, :]
            resized_era5_ = original_image[:,::4,:][:,:,::4]
            resized_bc_gfdl_lr.append(resized_era5_)
bc_gfdl_lr_64 = np.stack(resized_bc_gfdl_lr)
print("DM BC GFDL LR 64 shape:",bc_gfdl_lr_64.shape)

era5_hr_256.shape, hr_bc_gfld_sample.shape, bc_gfdl_sample.shape, bc_gfdl_lr_64.shape

# +
ssd = SpatialSpectralDensity_diff_res(   bc_gfdl_lr_64
                                        ,hr_bc_gfld_sample.unsqueeze(1).detach().cpu().numpy()
                                        ,era5_hr_256
                                        ,new_labels = ["BC GFDL LR", "DM BC GFDL HR", "ERA5 HR"])
ssd.run(num_times=None)

ssd.plot_psd(fname=f'/dss/dsshome1/0D/ge74xuf2/climate_diffusion/results/psd/era5_lr_vs_hr_vanilla.pdf'
             ,model_resolution=0.25,model_resolution_2=1)
# -
# # Run DM on whole data set to apply SR to era5 lr


do_save_sr_era5_dataset = False

# +
net = palette_model.load_pretrain_diffusion("models/DDPM_conditional/291_Diffusion_sr.pth")

print("MODEL USED TO CREATE BC GFDL: DDPM_conditional/291_Diffusion_sr.pth")
# -

if do_save_sr_era5_dataset == True:
    era5_p_1d_256 = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(stage='valid')
    era5_p_1d_256_ = era5_p_1d_256.data()

    dataloader_era5_lr_256_val = data.DataLoader(era5_p_1d_256_, batch_size=bs_valid,
                                                   shuffle=False, drop_last=True,
                                                   num_workers=2)

if do_save_sr_era5_dataset==True:
    output_tensors = []

    for b, el in enumerate(dataloader_era5_lr_256_val):
        era5_lr_256 = el.unsqueeze(1).to(device).float()

        sr_era5_hr_256, _ = net.restoration(era5_lr_256, sample_num=era5_lr_256.shape[0])
        output_tensors.append(sr_era5_hr_256)
        if b%10==0:
            print(b)
        #if b == 1:
        #    break

    sr_hr_era5_dataset = torch.cat(output_tensors, dim=0) 
    
    print(sr_hr_era5_dataset.shape)

if do_save_sr_era5_dataset==True:
    print("uncomment for saving")
    #torch.save(sr_hr_era5_dataset, 'data/SR_ERA5_HR_dataset.pth')

# ## eval SR on era5

loaded_dataset_SR_ERA5_HR = torch.load('data/SR_ERA5_HR_dataset.pth')
print("loaded data - shape:",loaded_dataset_SR_ERA5_HR.shape)

next(iter(loaded_dataset)).shape

batch_size_val = 20

# +
sr_hr_era5_dataset = SR_ERA5_HR_Dataset_256(hr_bc_gfdl_dataset)

dataloader_hr_bc_gfdl = data.DataLoader(sr_hr_era5_dataset.data(), batch_size=batch_size_val, shuffle=False,
                                        drop_last=True,num_workers=2)
DM_hr_era5_256 = next(iter(dataloader_hr_bc_gfdl)).unsqueeze(1).numpy()
DM_hr_era5_256.shape

# +
era5_p_1d_256 = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(stage='valid')
era5_p_1d_256_ = era5_p_1d_256.data()

dataloader_era5_lr_256_val = data.DataLoader(era5_p_1d_256_, batch_size=batch_size_val,
                                               shuffle=False, drop_last=True,
                                               num_workers=2)

for b, el in enumerate(dataloader_era5_lr_256_val):
        era5_lr_256 = el.unsqueeze(1).to(device).float().detach().cpu().numpy()
# -

era5_hr_256.shape, era5_lr_256.shape,DM_hr_era5_256.shape

resized_era5 = []
for i in range(era5_lr_256.shape[0]):
            original_image = era5_lr_256[i, :, :, :]
            resized_era5_ = original_image[:,::4,:][:,:,::4]
            resized_era5.append(resized_era5_)
era5_lr_64 = np.stack(resized_era5)
print("ERA5 LR 64 shape:",era5_lr_64.shape)

# +
ssd = SpatialSpectralDensity_diff_res( era5_lr_64
                                        ,era5_hr_256
                                        ,DM_hr_era5_256
                                        ,new_labels = ["ERA5 LR", "HR ERA5", "DM ERA5 SR"])
ssd.run(num_times=None)

ssd.plot_psd(fname=f'/dss/dsshome1/0D/ge74xuf2/climate_diffusion/results/psd/era5_lr_vs_hr_vanilla.pdf'
             ,model_resolution=0.25,model_resolution_2=1)
# -


