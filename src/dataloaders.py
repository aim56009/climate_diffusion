import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from skimage.transform import rescale
from scipy.ndimage import convolve
import torch.nn.functional as F

import os
new_directory = '/dss/dsshome1/0D/ge74xuf2/climate_diffusion'
os.chdir(new_directory)
os.getcwd()


class SLP_Dataset(torch.utils.data.Dataset):

    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
        self.splits = {
                "train":  (0, 6939),      ### missing training data 2014-2019
                "valid":  (6939, 8400),
                "test":   (6939, 10226),}
         
        self.era5_path = "data/model_data/SLP199201_202112.nc"    
        self.stage = stage
        self.era5 = self.load_era5_data()
        
        # 4. Normalize the data m=0, std=1      
        self.mean = np.mean(self.era5["slp"].values)
        self.std = np.std(self.era5["slp"].values)
        self.era5["slp"].values = (self.era5["slp"].values - self.mean) / self.std
        
        # 5. bring data to  [-1,1]      #### uncomment!!!
        min_value = self.era5["slp"].values.min()
        max_value = self.era5["slp"].values.max()
        self.era5["slp"].values = (self.era5["slp"].values - min_value) / (max_value - min_value)  
        self.era5["slp"].values = self.era5["slp"].values * 2 - 1 
        
        
        self.era5 = self.era5["slp"].sel(z=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        self.num_samples = len(self.era5.z.values)
        print("SLP_Dataset shape",self.era5.values.shape)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        return era5
        
    def get_mean_std(self):
        print("mean:",np.mean(self.era5.values))
        print("std:",np.std(self.era5.values))
        return np.mean(self.era5.values), np.std(self.era5.values)
        
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.era5.isel(z=index).values)[:self.size_x,:self.size_y]\
            .float().unsqueeze(0)
        return x 

    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array


class V850_Dataset(torch.utils.data.Dataset):

    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
        self.splits = {
                "train":  (0, 6939),      ### missing training data 2014-2019
                "valid":  (6939, 8400),
                "test":   (6939, 10226),}
         
        self.era5_path = "data/model_data/V850199201_202112.nc"    
        self.stage = stage
        self.era5 = self.load_era5_data()
        
        # 4. Normalize the data m=0, std=1      
        self.mean = np.mean(self.era5["v850"].values)
        self.std = np.std(self.era5["v850"].values)
        self.era5["v850"].values = (self.era5["v850"].values - self.mean) / self.std
        self.era5 = self.era5["v850"].sel(z=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        self.num_samples = len(self.era5.z.values)
        print("V850_Dataset shape",self.era5.values.shape)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        return era5
        
    def get_mean_std(self):
        print("mean:",np.mean(self.era5.values))
        print("std:",np.std(self.era5.values))
        return np.mean(self.era5.values), np.std(self.era5.values)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.era5.isel(z=index).values)[:self.size_x,:self.size_y]\
            .float().unsqueeze(0)
        return x 

    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array


class U850_Dataset(torch.utils.data.Dataset):

    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
        self.splits = {
                "train":  (0, 6939),      ### missing training data 2014-2019
                "valid":  (6939, 8400),
                "test":   (6939, 10226),}
         
        self.era5_path = "data/model_data/U850199201_202112.nc"    
        self.stage = stage
        self.era5 = self.load_era5_data()
        
        # 4. Normalize the data m=0, std=1      
        self.mean = np.mean(self.era5["u850"].values)
        self.std = np.std(self.era5["u850"].values)
        self.era5["u850"].values = (self.era5["u850"].values - self.mean) / self.std
        self.era5 = self.era5["u850"].sel(z=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("U850_Dataset shape",self.era5.values.shape)
        self.num_samples = len(self.era5.z.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        return era5
        
    def get_mean_std(self):
        print("mean:",np.mean(self.era5.values))
        print("std:",np.std(self.era5.values))
        return np.mean(self.era5.values), np.std(self.era5.values)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.era5.isel(z=index).values)[:self.size_x,:self.size_y]\
            .float().unsqueeze(0)
        return x 

    def __len__(self):
        return self.num_samples
    
    def size(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array


class T_Dataset(torch.utils.data.Dataset):

    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
        self.splits = {
                "train":  (0, 10958),
                "valid":  (6939, 8400),
                "test":   (6939, 10226),}
         
        self.era5_path = "data/model_data/SAT199201_202112.nc"    
        self.stage = stage
        self.era5 = self.load_era5_data()
        
        # 4. Normalize the data m=0, std=1      
        self.mean = np.mean(self.era5["sat"].values)
        self.std = np.std(self.era5["sat"].values)
        self.era5["sat"].values = (self.era5["sat"].values - self.mean) / self.std
        
        
        # 5. bring data to  [-1,1]      #### uncomment!!!
        min_value = self.era5["sat"].values.min()
        max_value = self.era5["sat"].values.max()
        self.era5["sat"].values = (self.era5["sat"].values - min_value) / (max_value - min_value)  
        self.era5["sat"].values = self.era5["sat"].values * 2 - 1
        
        
        
        self.era5 = self.era5["sat"].sel(z=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("T_Dataset shape",self.era5.values.shape)
        self.num_samples = len(self.era5.z.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        return era5
        
    def get_mean_std(self):
        print("mean:",np.mean(self.era5.values))
        print("std:",np.std(self.era5.values))
        return np.mean(self.era5.values), np.std(self.era5.values)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.era5.isel(z=index).values)[:self.size_x,:self.size_y]\
            .float().unsqueeze(0)
        return x 

    def __len__(self):
        return self.num_samples
    
    def size(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array


# # precipitation

class Antialiasing:
    def __init__(self):
        (x, y) = np.mgrid[-2:3, -2:3]
        self.kernel = np.exp(-0.5 * (x**2 + y**2) / (0.5**2))
        self.kernel /= self.kernel.sum()
        self.edge_factors = {}
        self.img_smooth = {}

    def __call__(self, img):
        img_shape = img.shape[-2:]
        if img_shape not in self.edge_factors:
            s = convolve(np.ones(img_shape, dtype=np.float32),
                         self.kernel, mode="constant")
            s = 1.0 / s
            self.edge_factors[img_shape] = s
        else:
            s = self.edge_factors[img_shape]

        if img_shape not in self.img_smooth:
            img_smooth = np.empty_like(img)
            self.img_smooth[img_shape] = img_smooth
        else:
            img_smooth = self.img_smooth[img_shape]

        for i in range(img.shape[0]):
            convolve(img[i], self.kernel, mode="constant", output=img_smooth[i])
            img_smooth[i] *= s

        return img_smooth


def dwd_rv_rainrate_transform(raw, threshold=0.1, fill_value=0.02, mean=-0.051, std=0.528,):
    antialiasing = Antialiasing()
    x = raw.copy()
    x[x < threshold] = fill_value
    x = np.log10(x, out=x)
    
    x = antialiasing(x)    #### cant be inverted -> does not matter ->for generated images just invert log trafo
    mu = np.mean(x)
    std = np.std(x)
    
    #print("mean", mu)
    #print("std",std)
    x -= mu        #for evaluation take the original data and compare to inv_log_trafo(no AA) generated 
    x /= std
    
    return x, mu, std


class ERA5_P_Dataset(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                "train":  (0, 6939),      ### (1992-2011) missing training data 2014-2019
                "valid":  (6939, 8400),   ## 01.01.2011 - 31.12.2014
                "test":   (8400, 10226),}

        self.era5_path = "data/model_data/P199201_202112.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
                
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(z=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("whole era5 dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.z.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).p*1000
            
        return era5
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(z=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x


class GFDL_P_Dataset_1_1(torch.utils.data.Dataset):
    """ from 2010-2014
    Data has lon-resolution: 1.25d, lat-resolution 1d , 
    lat, lon dimension: 26:90,216:267 (64,51) -> data spans the same region on the globe as era5
    data is interpolated to 64,64 pixels -> 1x1 degree
    
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                "train":  (364, 1825),   ## 01.01.2011 - 31.12.2014
                "valid":  (364, 1825),
                "test":   (364, 1825),}

        self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
                
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.isel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("size of whole dataset",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
            
        return era5
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5[index,26:90,216:267].values)
        flipped_x = torch.flip(x, dims=(0, 1))
        gfdl = torch.flip(flipped_x, dims=[1])
        
        gfdl_rescale = np.zeros((64, 64))
        gfdl_rescale = rescale(gfdl, scale=(1, 1.25), anti_aliasing=False)
        
        return torch.tensor(gfdl_rescale).float().unsqueeze(0)
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x


class ERA5_P_0_25_Dataset(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
        
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("whole era5 dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).tp*1000*24 #bc data was m/h 
                                                                    #-> averaged still m/h -> *1000/24 = mm/day
        return era5
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(time=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x

# %%capture
"""
gfdl_test = ERA5_P_0_25_Dataset(stage='train')
gfdl_test.get_mean_std()
gfdl_test = gfdl_test.data()

dataloader_gfdl_test = data.DataLoader(gfdl_test, batch_size=5, shuffle=False, drop_last=True,num_workers=2)

gfdl_valid_all = next(iter(dataloader_gfdl_test))

gfdl_valid_all = gfdl_valid_all.unsqueeze(dim=1)
print(gfdl_valid_all.shape)
"""


class ERA5_P_0_25_to_1_Dataset(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    
    # load 0.25d data (m/day)? -> trafo to mm/h ?
    # taking every 4th datapoint in x,y dimension -> 1d data
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
        
                
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("whole era5 dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).tp*1000*24 #bc data was m/h 
                                                                    #-> averaged still m/h -> *1000/24 = mm/day
        #print("era5", type(era5),era5.shape)
        resized_era5 = []

        for i in range(era5.shape[0]):
            original_image = era5[i, :, :]
            resized_era5_ = original_image[::4,:][:,::4]
            resized_era5.append(resized_era5_)

        resized_era5_data = np.stack(resized_era5)
        
        resized_era5_da = xr.DataArray(resized_era5_data, dims=('time', 'x', 'y'), coords={'time': era5.time})
        resized_era5_da.attrs = era5.attrs  # Copy attributes
        
    
        print("era5", type(era5),resized_era5_da.shape)
        return resized_era5_da #era5
    
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(time=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x


class ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    
    # load 0.25d data (m/day)? -> trafo to mm/h ?
    # taking every 4th datapoint in x,y dimension -> 1d data
    # just copy every datapoint 4 times -> 64x64 -> 256x256 pixel outputs
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
        
                
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("whole era5 dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).tp*1000*24 #bc data was m/h 
                                                                    #-> averaged still m/h -> *1000/24 = mm/day
        #print("era5", type(era5),era5.shape)
        resized_era5 = []

        for i in range(era5.shape[0]):
            original_image = era5[i, :, :]
            resized_era5_ = original_image[::4,:][:,::4]
            resized_era5.append(resized_era5_)

        resized_era5_data = np.stack(resized_era5)
        
        resized_era5_da = xr.DataArray(resized_era5_data, dims=('time', 'x', 'y'), coords={'time': era5.time})
        resized_era5_da.attrs = era5.attrs  # Copy attributes
        
    
        #print("era5", type(era5),resized_era5_da.shape)
        return resized_era5_da #era5
    
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(time=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0).unsqueeze(1)
        x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x


class GFDL_P_Dataset_to_256(torch.utils.data.Dataset):
    """ from 2010-2014
    Data has lon-resolution: 1.25d, lat-resolution 1d , 
    lat, lon dimension: 26:90,216:267 (64,51) -> data spans the same region on the globe as era5
    data is interpolated to 64,64 pixels -> 1x1 degree
    
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  (364, 1825),   ## 01.01.2011 - 31.12.2014
                "valid":  (364, 1825),
                "test":   (364, 1825),}

        self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
                
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.isel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("size of whole dataset",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
            
        return era5
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5[index,26:90,216:267].values)
        flipped_x = torch.flip(x, dims=(0, 1))
        gfdl = torch.flip(flipped_x, dims=[1])
        
        gfdl_rescale = np.zeros((64, 64))
        gfdl_rescale = rescale(gfdl, scale=(1, 1.25), anti_aliasing=False)
        
        x = torch.tensor(gfdl_rescale).float().unsqueeze(0).unsqueeze(0)
        #print("x.shape", x.shape)
        x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1)
        #print("x.shape", x.shape)
        
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x

# %%capture
"""
gfdl_test = GFDL_P_Dataset_to_256(stage='valid')
gfdl_test.get_mean_std()
gfdl_test = gfdl_test.data()

dataloader_gfdl_test = data.DataLoader(gfdl_test, batch_size=5, shuffle=False, drop_last=True,num_workers=2)

gfdl_valid_all = next(iter(dataloader_gfdl_test))

gfdl_valid_all = gfdl_valid_all.unsqueeze(dim=1)
print(gfdl_valid_all.shape)
"""

# +
import torchvision.transforms as transforms
from PIL import Image

class ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_gauss_blur(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    
    # load 0.25d data (m/day)? -> trafo to mm/h ?
    # taking every 4th datapoint in x,y dimension -> 1d data
    # just copy every datapoint 4 times -> 64x64 -> 256x256 pixel outputs
    ## add gausian blur to make era5 look more like gfdl
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
        
                
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("whole era5 dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).tp*1000*24 #bc data was m/h 
                                                                    #-> averaged still m/h -> *1000/24 = mm/day
        #print("era5", type(era5),era5.shape)
        resized_era5 = []

        for i in range(era5.shape[0]):
            original_image = era5[i, :, :]
            resized_era5_ = original_image[::4,:][:,::4]
            resized_era5.append(resized_era5_)

        resized_era5_data = np.stack(resized_era5)
        
        resized_era5_da = xr.DataArray(resized_era5_data, dims=('time', 'x', 'y'), coords={'time': era5.time})
        resized_era5_da.attrs = era5.attrs  # Copy attributes
        
    
        #print("era5", type(era5),resized_era5_da.shape)
        return resized_era5_da #era5
    
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(time=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0).unsqueeze(1)
        x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1)
        x = transforms.GaussianBlur(kernel_size=23, sigma=(2.0, 2.0))(x)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x

# +
import torchvision.transforms as transforms
from PIL import Image

class ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_blur_pix(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    
    # load 0.25d data (m/day)? -> trafo to mm/h ?
    # taking every 4th datapoint in x,y dimension -> 1d data
    # just copy every datapoint 4 times -> 64x64 -> 256x256 pixel outputs
    ## add gausian blur to make era5 look more like gfdl
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
        
                
        # 2. clip data to 0,128
        self.era5.values = np.clip(self.era5.values,0,128)
        
        # 3. quantize data 1/32 (maybe 1mm/h)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("whole era5 dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).tp*1000*24 #bc data was m/h 
                                                                    #-> averaged still m/h -> *1000/24 = mm/day
        #print("era5", type(era5),era5.shape)
        resized_era5 = []

        for i in range(era5.shape[0]):
            original_image = era5[i, :, :]
            resized_era5_ = original_image[::4,:][:,::4]
            resized_era5.append(resized_era5_)

        resized_era5_data = np.stack(resized_era5)
        
        resized_era5_da = xr.DataArray(resized_era5_data, dims=('time', 'x', 'y'), coords={'time': era5.time})
        resized_era5_da.attrs = era5.attrs  # Copy attributes
        
    
        #print("era5", type(era5),resized_era5_da.shape)
        return resized_era5_da #era5
    
    

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        # Calculate the number of quantization levels
        num_levels = int((max_value - min_value) / step_width)
        # Quantize the array
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(time=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0).unsqueeze(1)
        """bring 1 degree image with 64x64 pixels to 256x256 pixel"""
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        """make image more blured"""
        sig = 2.0
        x = transforms.GaussianBlur(kernel_size=23, sigma=(sig, sig))(x)
        """make image more pixelated"""
        pixel_size = 3  # Adjust this value to control the pixelation level      ### 4 seems to be close to gfdl
        # Resize the image tensor to a smaller size
        resized_image = F.interpolate(x, size=(256 // pixel_size, 256 // pixel_size), mode='nearest')
        # Upscale the resized image back to the original size
        x = F.interpolate(resized_image, size=(256, 256), mode='nearest').squeeze(1)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x
# -

# %%capture
"""
gfdl_test = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256_blur_pix(stage='valid')
gfdl_test.get_mean_std()
gfdl_test = gfdl_test.data()

dataloader_gfdl_test = data.DataLoader(gfdl_test, batch_size=5, shuffle=False, drop_last=True,num_workers=2)

gfdl_valid_all = next(iter(dataloader_gfdl_test))

gfdl_valid_all = gfdl_valid_all.unsqueeze(dim=1)
print(gfdl_valid_all.shape)
plt.imshow(gfdl_valid_all[0,0,:,:])
"""

# +
compare = False

if compare == True:
    era5_256_comp = ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(stage='valid')
    loader_era5_256_comp = data.DataLoader(era5_256_comp.data(), batch_size=5, shuffle=False, drop_last=True,num_workers=2)
    plt.imshow(next(iter(loader_era5_256_comp))[0,:,:])


# -

class BC_GFDL_Dataset(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                "train":  (364, 1825),   ## 01.01.2011 - 31.12.2014
                "valid":  (364, 1825),
                "test":   (364, 1825),}

        #self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        self.era5_path = "data/model_data/P199201_202112.nc"
        self.stage = stage
        self.era5 = self.load_era5_data()
        self.era5.values = np.clip(self.era5.values,0,128)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        self.min_value = self.era5.values.min()
        self.max_value = self.era5.values.max()

        
        self.bc_gfdl = torch.load('data/bias_corr_gfdl_dataset.pth')
        print("size of whole dataset",self.bc_gfdl.shape)
        self.num_samples = self.bc_gfdl.shape[0]
        

    def load_era5_data(self):
        #era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).p*1000
        return era5

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        num_levels = int((max_value - min_value) / step_width)
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        return self.bc_gfdl[index]
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x


class BC_GFDL_Dataset_256(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  (364, 1825),   ## 01.01.2011 - 31.12.2014
                "valid":  (364, 1825),
                "test":   (364, 1825),}

        #self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        self.era5_path = "data/model_data/P199201_202112.nc"
        self.stage = stage
        self.era5 = self.load_era5_data()
        self.era5.values = np.clip(self.era5.values,0,128)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        self.min_value = self.era5.values.min()
        self.max_value = self.era5.values.max()

        
        self.bc_gfdl = torch.load('data/bias_corr_gfdl_dataset.pth')
        print("size of whole dataset",self.bc_gfdl.shape)
        self.num_samples = self.bc_gfdl.shape[0]
        

    def load_era5_data(self):
        #era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).p*1000
        return era5

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        num_levels = int((max_value - min_value) / step_width)
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = self.bc_gfdl[index].unsqueeze(1)
        x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1)
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x


class UNET_VALID_Dataset_64(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64 #256   #uncomment if you want to upsample to 256
        self.size_y = 64 #256   #uncomment if you want to upsample to 256
    
        self.splits = {
                "train":  (364, 1825),   ## 01.01.2011 - 31.12.2014
                "valid":  (364, 1825),
                "test":   (364, 1825),}

        #self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        self.era5_path = "data/model_data/P199201_202112.nc"
        self.stage = stage
        self.era5 = self.load_era5_data()
        self.era5.values = np.clip(self.era5.values,0,128)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        self.min_value = self.era5.values.min()
        self.max_value = self.era5.values.max()

        
        self.bc_gfdl = torch.load('data/UNET_era5_lr_64_dataset.pth')
        print("size of whole dataset",self.bc_gfdl.shape)
        self.num_samples = self.bc_gfdl.shape[0]
        

    def load_era5_data(self):
        #era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).p*1000
        return era5

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        num_levels = int((max_value - min_value) / step_width)
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = self.bc_gfdl[index].unsqueeze(1)                            #uncomment if you want to upsample to 256
        #x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1) #uncomment if you want to upsample to 256
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].detach().cpu().numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x

# +
#bc_gfdl_dataset = UNET_VALID_Dataset_64("valid")
#bc_gfdl_dataset_ = bc_gfdl_dataset.data()


#dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=10, shuffle=False, drop_last=True,num_workers=2)
#next(iter(dataloader_bc_gfdl)).shape
# -

class BC_UNET_VALID_Dataset_64(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64 #256   #uncomment if you want to upsample to 256
        self.size_y = 64 #256   #uncomment if you want to upsample to 256
    
        self.splits = {
                "train":  (364, 1825),   ## 01.01.2011 - 31.12.2014
                "valid":  (364, 1825),
                "test":   (364, 1825),}

        #self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        self.era5_path = "data/model_data/P199201_202112.nc"
        self.stage = stage
        self.era5 = self.load_era5_data()
        self.era5.values = np.clip(self.era5.values,0,128)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        self.min_value = self.era5.values.min()
        self.max_value = self.era5.values.max()

        
        self.bc_gfdl = torch.load('data/bias_corr_unet_val_dataset.pth')
        print("size of whole dataset",self.bc_gfdl.shape)
        self.num_samples = self.bc_gfdl.shape[0]
        

    def load_era5_data(self):
        #era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).p*1000
        return era5

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        num_levels = int((max_value - min_value) / step_width)
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = self.bc_gfdl[index].unsqueeze(1)                            #uncomment if you want to upsample to 256
        #x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1) #uncomment if you want to upsample to 256
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x

# +
#bc_gfdl_dataset = BC_ERA5_VALID_Dataset_256("valid")
#bc_gfdl_dataset_ = bc_gfdl_dataset.data()


#dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=10, shuffle=False, drop_last=True,num_workers=2)
#next(iter(dataloader_bc_gfdl)).shape
# -

class SR_BC_GFDL_Dataset_256(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  (364, 1825),   ## 01.01.2011 - 31.12.2014
                "valid":  (364, 1825),
                "test":   (364, 1825),}

        #self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        self.era5_path = "data/model_data/P199201_202112.nc"
        self.stage = stage
        self.era5 = self.load_era5_data()
        self.era5.values = np.clip(self.era5.values,0,128)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        self.min_value = self.era5.values.min()
        self.max_value = self.era5.values.max()

        
        self.hr_bc_gfdl = torch.load('data/HR_BC_GFDL_dataset.pth')
        print("size of whole dataset",self.hr_bc_gfdl.shape)
        self.num_samples = self.hr_bc_gfdl.shape[0]
        

    def load_era5_data(self):
        #era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).p*1000
        return era5

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        num_levels = int((max_value - min_value) / step_width)
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = self.hr_bc_gfdl[index]
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x


class SR_ERA5_HR_Dataset_256(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  (0, 6939),      ### (1992-2011) missing training data 2014-2019
                "valid":  (6939, 8400),   ## 01.01.2011 - 31.12.2014
                "test":   (8400, 10226),}

        #self.era5_path = "data/model_data/pr_day_GFDL-ESM4_esm-hist_r1i1p1f1_gr1_20100101-20141231.nc"
        self.era5_path = "data/model_data/P199201_202112.nc"
        self.stage = stage
        self.era5 = self.load_era5_data()
        self.era5.values = np.clip(self.era5.values,0,128)
        self.era5.values = self.quantize_array(self.era5.values)    ###comment in
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform(self.era5.values)
        self.min_value = self.era5.values.min()
        self.max_value = self.era5.values.max()

        
        self.hr_bc_gfdl = torch.load('data/SR_ERA5_HR_dataset.pth')
        print("size of whole dataset",self.hr_bc_gfdl.shape)
        self.num_samples = self.hr_bc_gfdl.shape[0]
        

    def load_era5_data(self):
        #era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).p*1000
        return era5

    def quantize_array(self,array, min_value=0, max_value=128, step_width=1/32): # was 1/32
        num_levels = int((max_value - min_value) / step_width)
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
 
        return quantized_array
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = self.hr_bc_gfdl[index]
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x

# %%capture
"""
bc_gfdl_dataset = SR_BC_GFDL_Dataset_256("valid")
bc_gfdl_dataset_ = bc_gfdl_dataset.data()

dataloader_bc_gfdl = data.DataLoader(bc_gfdl_dataset_, batch_size=10, shuffle=False, drop_last=True,num_workers=2)
"""

# +
plot_bc_gfdl_sample=False

if plot_bc_gfdl_sample==True:
    bc_gfld_sample = next(iter(dataloader_bc_gfdl))
    print("batch size:",bc_gfld_sample.shape)
    plt.imshow(bc_gfld_sample[0,:,:].cpu())
    plt.show()
# -

# # test inverting the dwd trafo

do_testing = False

if do_testing==True:
    from utils import *
    era5_test_load = xr.open_dataset("data/model_data/era5_daymean_1992_2014.nc", cache=True, chunks=None).tp*1000*24

if do_testing==True:
    def quantize_array(array, min_value=0, max_value=128, step_width=1/32): 
        num_levels = int((max_value - min_value) / step_width)
        quantized_array = np.round((array - min_value) / step_width) * step_width + min_value
        return quantized_array

    def complete_pr_processing(x):
        x = np.clip(era5_test_load.values,0,128)
        x = quantize_array(x)
        x, mu, std = dwd_rv_rainrate_transform(x)
        min_value = x.min()
        max_value = x.max()
        x = (x - min_value) / (max_value - min_value)  
        x = x * 2 - 1 
        print("mean", mu)
        print("std", std)
        print("min_value",min_value)
        print("max_value",max_value)
        return x

    new_data = complete_pr_processing(era5_test_load)

if do_testing==True:
    def inverse_dwd_rv_rainrate_transform(transformed_data, mean= -0.29383737, std=0.96417475, 
                                          min_value=-1.4573424, max_value=2.4902616):
        x = transformed_data.copy()
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        # Reset values below the threshold
        #x[x < threshold] = 0    .. not really possible
        return x

    data_trafo_and_back = inverse_dwd_rv_rainrate_transform(new_data)

if do_testing==True:
    histograms_three_np(original=data_trafo_and_back,
                        generated=np.clip(era5_test_load.values,0,128),
                        label=data_trafo_and_back
                        ,xlim_end=None, label_name=["raw","quantized raw","after trafo and back"],var="p")
    print("not 100% identiacal bc of AA operation in trafo")


