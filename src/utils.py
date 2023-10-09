import matplotlib.pyplot as plt
import numpy as np
import torch


def histograms(original, generated, log=True, xlim_end=None, var="t"):
    _, ax = plt.subplots()

    original = original.cpu().flatten()
    generated = generated.cpu().flatten()

    _ = plt.hist(original,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label='Data')


    _ = plt.hist(generated,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label='Generated')
    
    if var=="t":
        plt.xlabel('Mean Temperature [K]')
    else:
        plt.xlabel('Mean Precipitation [mm/d]')
    plt.ylabel('Histogram')
    if xlim_end:
        plt.xlim(0, xlim_end)
    plt.grid()
    plt.legend()
    plt.show()


def histograms_three(original, generated, label, log=True, xlim_end=None, var="t", label_name=['Original', 'Generated', 'Unet']):
    _, ax = plt.subplots()

    original = original.cpu().flatten()
    generated = generated.cpu().flatten()
    label = label.cpu().flatten()

    _ = plt.hist(original,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[0])


    _ = plt.hist(generated,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[1])
    
    
    _ = plt.hist(label,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[2])
    
    
    if var=="t":
        plt.xlabel('Mean Temperature [K]')
    else:
        plt.xlabel('Mean Precipitation [mm/d]')
    plt.ylabel('Histogram')
    if xlim_end:
        plt.xlim(0, xlim_end)
    plt.grid()
    plt.legend()
    plt.show()


def histograms_three_np(original, generated, label, log=True, xlim_end=None, var="t", label_name=['Original', 'Generated', 'Unet']):
    _, ax = plt.subplots()

    original = original.flatten()
    generated = generated.flatten()
    label = label.flatten()

    _ = plt.hist(original,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[0])


    _ = plt.hist(generated,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[1])
    
    
    _ = plt.hist(label,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[2])
    
    
    if var=="t":
        plt.xlabel('Mean Temperature [K]')
    else:
        plt.xlabel('Mean Precipitation [mm/d]')
    plt.ylabel('Histogram')
    if xlim_end:
        plt.xlim(0, xlim_end)
    plt.grid()
    plt.legend()
    plt.show()


def latitudinal_mean(original, generated, var="t"): 
    original = original.cpu().mean(dim=(0, 1, 2))
    generated = generated.cpu().mean(dim=(0, 1, 2))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label='original')
    ax.plot(latitudes, generated, label='generated')
    ax.set_xlabel('Latitude')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def latitudinal_mean_three(original, generated, label, var="t", label_name=['Original', 'Generated', 'Unet']): 
    original = original.cpu().mean(dim=(0, 1, 2))
    generated = generated.cpu().mean(dim=(0, 1, 2))
    label = label.cpu().mean(dim=(0, 1, 2))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel('Latitude')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def latitudinal_mean_three_np(original, generated, label, var="t", label_name=['Original', 'Generated', 'Unet']): 
    original = np.mean(original, axis=(0, 1, 2))
    generated = np.mean(generated, axis=(0, 1, 2))
    label = np.mean(label, axis=(0, 1, 2))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel('Latitude')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def plot_images_no_lab(images):
    #images = images.cpu().detach().numpy()
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu().detach().numpy())
    num_images = len(images)
    tick_positions = [(i + 0.5) * images[0].shape[-1] for i in range(num_images)]
    plt.xticks(tick_positions)
    plt.yticks([])
    plt.show()


"""
import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
"""

"""
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
"""

"""
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
"""

"""
def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
"""

"""
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
"""


class SpatialSpectralDensity_4_diff_res():
    """
    1 & 2 argument have a different resolution than argument 2&3 
    """
    def __init__(self, original, generated, label, comparison=None, 
                 new_labels = ["original","generated", "UNet", "unet"]):
        self.original = original
        self.generated = generated  
        self.label = label  
        self.new_label_ori = new_labels[0]
        self.new_label_gen = new_labels[1]
        self.new_label_lab = new_labels[2]
        self.comparison = comparison
        self.new_label_comp = new_labels[3]
   
    def compute_mean_spectral_density(self, data):
        
        num_frequencies = np.max(((data.shape[3]),(data.shape[3])))/2
        mean_spectral_density = np.zeros(int(num_frequencies))
        num_times = int(len(self.original[:,0,0,0]))

        
        for t in range(num_times):
            tmp = data[t,0,:,:]
            psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)            
            mean_spectral_density += psd
        mean_spectral_density /= num_times
        
        return mean_spectral_density, freq
    
    def run(self, num_times=None, timestamp=None):
        self.num_times = num_times
        self.timestamp = timestamp
        self.original_psd, self.freq_lr = self.compute_mean_spectral_density(self.original)
        self.generated_psd, self.freq_lr = self.compute_mean_spectral_density(self.generated)
        self.label_psd, self.freq = self.compute_mean_spectral_density(self.label)
        self.comp_psd, self.freq = self.compute_mean_spectral_density(self.comparison)
    
    
    def plot_psd(self, axis=None, fname=None, fontsize=None, linewidth=None, model_resolution=0.5,
                 model_resolution_2=0.5):
        
        if axis is None: 
            _, ax = plt.subplots(figsize=(7,6))
        else:
            ax = axis
        plt.rcParams.update({'font.size': 12})
        """
        resolution example: 1 degree / pixel  = ?km?
        equatorial circumference is approximately 40,075km
        40,075 kilometers / 360 degrees ≈ 111.32 km per degree
        -> 1 degree ≈ 111.32 km/degree
        
        for 0.25 res:
        1 pixel = 0.25 degrees/pixel
        1 degree ≈ 40,075 km / 360 degrees ≈ 111.32 km / degree
        0.25 degrees/pixel * 111.32 kilometers/degree ≈ 27.83
        """
        x_vals = 1/self.freq*model_resolution*111 /2 #    why / 2 -> prbl. bc there is long, lat 
        x_vals_lr = 1/self.freq_lr*model_resolution_2*111/2
        
        print("len", len(self.freq), len(self.freq_lr))
        
        print("hr smallest km:",x_vals.min())
        print("lr smallest km:", x_vals_lr.min())


        #x_vals = np.flip(1/self.freq*model_resolution*111/2)
        ax.plot(x_vals_lr, self.original_psd, label= self.new_label_ori, color='k', linewidth=linewidth)
        ax.plot(x_vals_lr, self.generated_psd , label= self.new_label_gen, color='r', linewidth=linewidth)
        ax.plot(x_vals, self.label_psd , label= self.new_label_lab, color='g', linewidth=linewidth)
        ax.plot(x_vals, self.comp_psd , label= self.new_label_comp, color='b', linewidth=linewidth)
            
        ax.legend(loc='lower right', fontsize=fontsize)
        #ax.set_yscale('log', basey=2)
        #ax.set_xscale('log', basex=2)
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        #ax.set_xticks([2**9, 2**10, 2**11, 2**12, 2**13])
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid()
        
        ax.set_xlabel(r'k [km]', fontsize=fontsize)
        ax.set_ylabel('PSD [a.u]', fontsize=fontsize)
        ax.set_title("PSD")

        if fname is not None:
            plt.savefig(fname, format='pdf', bbox_inches='tight')


