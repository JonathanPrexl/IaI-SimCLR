import os
import glob
import numpy as np
import json

import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from enum import Enum, unique
from abc import ABC, abstractmethod
import hydra

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

from augmentations import TransformsPostivePair

from utils import load_s1_image, load_s2_image, load_lc
from utils import load_s1_image_BioMasters, load_s2_image_BioMasters

from utils import preprocess_s1, preprocess_s2, preprocess_lc

class BaseDataloader(Dataset):

    def __init__(self, config, trainvaltestkey):

        self.topdir_dataset = config.dataloader.topdir_dataset
        self.trainvaltestkey = trainvaltestkey
        
        # load the file with the realive locations of the patches
        relative_locations_file = config.dataloader.relative_locations
        with open(relative_locations_file, "r") as f:
            self.relative_locations = json.load(f)[trainvaltestkey]

        # restrict the number of sampels if so specified
        # in the config file
        if trainvaltestkey == "train":
            if config.restrict_train_data != -1:
                self.relative_locations = self.relative_locations[:config.restrict_train_data]
        elif trainvaltestkey == "val":
            if config.restrict_val_data != -1:
                self.relative_locations = self.relative_locations[:config.restrict_val_data]
        else:
            raise ValueError("Incorect trainvalkey")


        print(f"In total {len(self.relative_locations)} for {trainvaltestkey}")

    @abstractmethod
    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.relative_locations)

class Sen12MS(BaseDataloader):

    """ 
    returns S1, S2 and Annotation Tensors in this order
    """

    def __init__(self, config, trainvaltestkey, **kwargs):

        self.augmentations_inter = hydra.utils.instantiate(config.augmentations_inter)
        self.augmentations_S1 = hydra.utils.instantiate(config.augmentations_S1)
        self.augmentations_S2 = hydra.utils.instantiate(config.augmentations_S2)

        super().__init__(config, trainvaltestkey)   

        # locate all sampels that are close to each other
        # aka from the same scene
        self.neighbourSampels = {"ROIs1158_spring":{},
                                 "ROIs1970_fall":{},
                                 "ROIs1868_summer":{},
                                 "ROIs2017_winter":{}}

        for sidx, sample in enumerate(self.relative_locations):
            season = sample["s1"].split(os.sep)[0]
            patchID = sample["s1"].split(os.sep)[1] # aka s1_31
            patchID = int( patchID.split("_")[-1] ) # --> 31

            if patchID not in self.neighbourSampels[season].keys():
                self.neighbourSampels[season][patchID] = []
            else:
                self.neighbourSampels[season][patchID].append(sidx)

        pass

    def neighbourSampels(self):
        return self.neighbourSampels


    def __getitem__(self, i):
    
        s1_loc = os.path.join(self.topdir_dataset,
                              self.relative_locations[i]["s1"])

        s2_loc = os.path.join(self.topdir_dataset,
                              self.relative_locations[i]["s2"])

        lc_loc = os.path.join(self.topdir_dataset,
                              self.relative_locations[i]["landcover"])

        # read s1 image from disk
        data_s1 = load_s1_image(s1_loc).astype("float32")
        data_s1 = preprocess_s1(data_s1)
        data_s1 = torch.Tensor(data_s1.to_numpy())

        # read s2 image from disk
        # only take the 10 bands with 10 or 20m GSD
        bands_to_load = ["B02","B03","B04","B05","B06","B07","B08","B08A","B11","B12"]
        data_s2 = load_s2_image(s2_loc).loc[bands_to_load].astype("float32")
        data_s2 = preprocess_s2(data_s2)
        data_s2 = torch.Tensor(data_s2.to_numpy())

        # read annotation from disk
        data_lc = load_lc(lc_loc).astype("long")
        data_lc = preprocess_lc(data_lc)
        data_lc = torch.Tensor(data_lc)

        s1 = self.augmentations_inter(data_s1)
        s1d = self.augmentations_S1(data_s1)
        s1dd = self.augmentations_S1(data_s1)

        s2 = self.augmentations_inter(data_s2)
        s2d = self.augmentations_S2(data_s2)
        s2dd = self.augmentations_S2(data_s2)

        return {"s1":s1,"s1d":s1d,"s1dd":s1dd,
                "s2":s2,"s2d":s2d,"s2dd":s2dd,
                "lc":data_lc,
                "id":self.relative_locations[i]["s2"]}

if __name__ == "__main__":

    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./configs/eval_BioMasters.yaml')

    dl = BioMasters(cfg,"train")
    batch = dl.__getitem__(0)

    pass