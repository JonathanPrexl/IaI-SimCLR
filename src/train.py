# basic libarys
import os
import glob
import numpy as np
from datetime import datetime
import json
import csv
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
import hydra

# geo stuff
import rasterio

# importing learning stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchmetrics import MetricTracker, MetricCollection
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

# importing own modules
from dataloader import Sen12MS
from losses import NT_Xent_SingGPU as NTXent
from model import ResNetEncoder



from utils import weight_histograms, s2toRGB

# =============================== Abstract Baseclass of Trainer ==================================

class BaseTrainer(ABC):
    
    def __init__(self,config):
        
        # some variables we need
        self.config = config
        self.globalstep = 0
        self.loss = 0

        if config.gpu_idx != "cpu":
            self.cuda = True
        else:
            self.cuda = False

        # seeding
        np.random.seed(config.seed)
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
        else:
            torch.manual_seed(config.seed)

        # make folders 
        date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.savepath = os.path.join(config.outputpath, config.experimentname, date_time)
        self.checkpoint_dir = os.path.join(self.savepath,"model_checkpoints")
        os.makedirs(self.savepath, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        # store the config file that has been used
        # to make it more traceable
        with open(os.path.join(self.savepath,"used_parameters.json"), 'w') as f:
            json.dump(OmegaConf.to_container(config), f) 

        # sice we have constant imput size in each
        # iteration we can set up benchmark mode
        # (its just a bit faster thats it...)
        torch.backends.cudnn.benchmark = True

        # loading datasets 
        dataset_class = hydra.utils.instantiate(config.dataloader)
        self.train_set = dataset_class(config, trainvaltestkey="train")
        self.val_set = dataset_class(config, trainvaltestkey="val")


        batchSampler = hydra.utils.instantiate(config.batchsampler)

        self.batchSampler_train = batchSampler(neighbourSampels = self.train_set.neighbourSampels,
                                               lenDS = len(self.train_set),
                                               batch_size=config.dataloader.batch_size)
        
        self.batchSampler_val = batchSampler(neighbourSampels = self.val_set.neighbourSampels,
                                             lenDS = len(self.val_set),
                                             batch_size=config.dataloader.batch_size)


        self.training_data_loader = DataLoader(dataset=self.train_set,
                                               num_workers=config.dataloader.threads,
                                               batch_sampler=self.batchSampler_train)

        self.val_data_loader = DataLoader(dataset=self.val_set,
                                          num_workers=config.dataloader.threads,
                                          batch_sampler=self.batchSampler_val)


        # if we start from scratch
        if config.resume == False:

            # setup the model loss and optimizer
            # this is chaning for every experiment
            self._setup_model_loss_optimizer(config)

            # setup lr schedulert
            if not self.config.lrscheduler._target_ == None: 
                self.scheduler = hydra.utils.instantiate(config.lrscheduler, optimizer=self.optimizer)
            else:
                self.scheduler = None

        elif config.resume == True:
            
            if not os.path.isfile(config.resumeCheckpoint):
                raise ValueError(f"checkpointfile {config.resumeCheckpoint} does not exist")

            # setup the model loss and optimizer
            # this is chaning for every experiment
            self._setup_model_loss_optimizer(config, loadCPT=True)

            # setup lr schedulert
            if not self.config.lrscheduler._target_ == None: 
                self.scheduler = hydra.utils.instantiate(config.lrscheduler, optimizer=self.optimizer)
            else:
                self.scheduler = None

            if not self.scheduler == None:
                raise ValueError("Checkpoint loading with lr scheduler not supported yet^^")


        else:
            raise ValueError("")


        # setup tensorboard
        self.TB_writer = SummaryWriter(log_dir=os.path.join(self.savepath,"logs"))
        self.TB_writer.add_text("Parameters",str(config))

        # save the config if we need it somehwere else as well
        self.config = config
        
        # when to validate:
        # to make it comparable we dont do it after x batches rather after x sampels
        self.nextValidationstep = self.config.validation_every_N_sampels
        self.best_metric_saveCrit = 10000
            
    def fit(self):
        
        self.current_epoch = 1

        if self.config.resume:
            self.current_epoch = self.init_epoch
            self.globalstep = self.init_global_step
            self.loss = self.init_loss
            if not self.config.nEpochs > self.current_epoch:
                raise ValueError(f"Model checkpoint has been trained for {self.current_epoch} and final number of epochs is {self.config.nEpochs}")
        
        for epoch in range(self.current_epoch, self.config.nEpochs + 1):

            self._train_one_epoch()

            if not self.config.validate_after_every_n_epoch == -1:
                if epoch % self.config.validate_after_every_n_epoch == 0:
                    self.model.eval() # todo: be carefull projection head
                    self._validate()
                    self.model.train()

            # if lr sceduler is active
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    # otherwise we only update if we have not hit T_max
                    # otherwise we leave constant... if we would not do that it would rise again
                    # -2 so we hit appox eta_min...
                    if self.current_epoch - 2 < self.config.lrscheduler.T_max:
                         self.scheduler.step()
                else:
                    self.scheduler.step()

            if self.current_epoch in self.config.special_save_nEpoch:
                self._save_checkpoint(nameoverwrite = f"special_save_E{self.current_epoch}")

            self.TB_writer.add_scalar(f"lr", self.optimizer.param_groups[0]["lr"], global_step=self.globalstep)
            self.TB_writer.add_scalar(f"lr/over_epoch", self.optimizer.param_groups[0]["lr"], global_step=self.current_epoch)
            
            self.current_epoch += 1

        self.TB_writer.flush()

        return None
   
    def _train_one_epoch(self):
         
        self.model.train()

        pbar_train = tqdm(total=self.batchSampler_train.numBatches, desc=f"EPOCH: {self.current_epoch}",leave=False)
        
        for batch_idx, batch in enumerate(self.training_data_loader, start=1): 
            
            self._train_one_batch(batch)
            
            if batch_idx == 1:
                
                # once a epoch show some example imgs in the
                # tensorboard

                for dashes in ["","d","dd"]:
                    firstS1 = batch["s1"+dashes][:10]
                    firstS2 = batch["s2"+dashes][:10]
                    figure, loax = plt.subplots(2,10,figsize=(20,4))
                    for i in range(10):
                        loax[0,i].imshow( s2toRGB(firstS2[i]) ) 
                        loax[1,i].imshow( firstS1[i,0] ) 
                        loax[0,i].axis("off")
                        loax[1,i].axis("off")
                    self.TB_writer.add_figure(f"exampleInput/dashes_{dashes}", figure, global_step=self.globalstep)
                    plt.close()

            if batch_idx % 5 == 0:
                pbar_train.set_description("Epoch: {}".format(self.current_epoch))
            
            if not self.config.validation_every_N_sampels == -1:
                if (self.globalstep * self.config.dataloader.batch_size) >= self.nextValidationstep:
                    
                    self.model.eval()
                    self._validate()                
                    # set when to validate next time
                    self.nextValidationstep += self.config.validation_every_N_sampels
                    # set back to train mode
                    self.model.train()
            
            pbar_train.update()
        pbar_train.close()        
        
        return None

    def _setup_model_loss_optimizer(self, config, loadCPT=False):

        # loading model
        self.model: torch.nn.Module = hydra.utils.instantiate(config.model)

        if loadCPT:
            cpt = torch.load(config.resumeCheckpoint)
            self.model.load_state_dict(cpt["model_state_dict"])

        summary(self.model)

        # setup lossfunction
        assert config.loss.batch_size == config.dataloader.batch_size, "BS should match"
        self.lossfunction = hydra.utils.instantiate(config.loss)



        # if you have a gpu we
        # shift all on the GPU
        if self.cuda:
            self.model = self.model.cuda()
            self.lossfunction = self.lossfunction.cuda()

        # set up the optimizer
        self.optimizer: torch.optim.optimizer.Optimizer = hydra.utils.instantiate(config.optimizer,
                                                                                  params=self.model.parameters())
                                        
        if loadCPT:
            self.optimizer.load_state_dict(cpt["optimizer_state_dict"])


        # remember last epoch step and loss
        if loadCPT:
            self.init_epoch = cpt["epoch"]
            self.init_global_step = cpt["global_step"]
            self.init_loss = cpt["loss"]


    @abstractmethod
    def _train_one_batch(self, batch):

        """
        the core of the training... this you have to specify for each
        trainer seperately
        """
        pass

    @abstractmethod
    def _validate(self, batch):

        """
        the core of the training... this you have to specify for each
        trainer seperately
        """
        pass
        
    def _save_checkpoint(self,nameoverwrite = ""):
        
        # only delete if nameoverwrite is "" so that
        # at last epoch we dont delte the inbetween checkpoint
        if nameoverwrite == "":
            all_models_there = glob.glob(os.path.join(self.checkpoint_dir,"checkpoint*.pt"))
            # there should be none or one model
            if not len(all_models_there) in [0,1]:
                raise ValueError(f"There is more then one model in the checkpoint dir ({len(self.checkpoint_dir)})... seems wrong")
            else:
                for model in all_models_there:
                    os.remove(model)
        
        if nameoverwrite == "":
            outputloc =  os.path.join(self.checkpoint_dir,f"checkpoint_{self.current_epoch}_{self.globalstep}_{self.best_metric_saveCrit}.pt")
        else:
            outputloc =  os.path.join(self.checkpoint_dir,f"{nameoverwrite}.pt")

        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.globalstep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            },
           outputloc)
          
        return None
              
    def finalize(self):

        # save hyperparameters
        self.TB_writer.add_hparams(
                    {"lr": self.config.optimizer.lr,
                     "bsize": self.config.dataloader.batch_size,
                    },
                    {
                    "hparam/IoU": self.best_metric_saveCrit,
                    },
                    run_name="hparams"
                )

        self.TB_writer.close()
        
        self._save_checkpoint("state_at_finalize")
        
        return None

class DualSimCLR(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
                       
    def _train_one_batch(self, batch):
        
        s1 = batch["s1"]
        s2 = batch["s2"]
        
        if self.cuda:
            s1 = s1.cuda()
            s2 = s2.cuda()

        h_s1, h_s2, z_s1, z_s2 = self.model(s1, s2)

        self.loss, self.correct_order = self.lossfunction(z_s1, z_s2)

        self.optimizer.zero_grad()

        self.loss.backward()

        self.optimizer.step()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/loss_inter", self.loss, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/correct_order_inter", self.correct_order, global_step=self.globalstep)
        
        return None
    
    def _validate(self):
        
        list_of_val_losses = []

        with torch.no_grad():
            
            pbar_val = tqdm(total=self.batchSampler_val.numBatches, desc=f"EPOCH: {self.current_epoch}",leave=False)
            pbar_val.set_description("Validation")

            for batch_idx, batch in enumerate(self.val_data_loader):
 
                # prepare data
                s1 = batch["s1"]
                s2 = batch["s2"]        

                if self.cuda:
                    s1 = s1.cuda()
                    s2 = s2.cuda()

                h_s1, h_s2, z_s1, z_s2 = self.model(s1, s2)

                list_of_val_losses.append(self.lossfunction(z_s1,z_s2))

                pbar_val.update()
                
            pbar_val.close()   

            # write to tensorboard
            loss_on_val_set = torch.mean(torch.Tensor(list_of_val_losses))
            if not self.config.validation_every_N_sampels == -1:
                self.TB_writer.add_scalar(f"val/loss_inter", loss_on_val_set, global_step=self.globalstep)
            if not self.config.validate_after_every_n_epoch == -1:
                self.TB_writer.add_scalar(f"val/epoch_loss_inter", loss_on_val_set, global_step=self.current_epoch)

            if loss_on_val_set < self.best_metric_saveCrit:
                self.best_metric_saveCrit = loss_on_val_set
                self._save_checkpoint()

        return None

class IaISimCLR(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
                                   
    def _train_one_batch(self, batch):
        
        s1 = batch["s1"]
        s1d = batch["s1d"]
        s1dd = batch["s1dd"]

        s2 = batch["s2"]
        s2d = batch["s2d"]
        s2dd = batch["s2dd"]
        
        if self.cuda:

            s1 = s1.cuda()
            s1d = s1d.cuda()
            s1dd = s1dd.cuda()

            s2 = s2.cuda()
            s2d = s2d.cuda()
            s2dd = s2dd.cuda()

        # try to save memory
        h_s1, h_s2, z_s1, z_s2, z_s1d, z_s2d = self.model(s1, s2)
        self.loss_inter, self.correct_order_inter = self.lossfunction(z_s1, z_s2) 
        self.loss_inter *= self.config.lossweights.lambda_inter
        h_s1, h_s2, z_s1, z_s2, z_s1d, z_s2d = self.model(s1d, s2d)
        h_s1, h_s2, z_s1, z_s2, z_s1dd, z_s2dd = self.model(s1dd, s2dd) 
        self.loss_intra_s1, self.correct_order_intra_s1 = self.lossfunction(z_s1d, z_s1dd) 
        self.loss_intra_s1 *= self.config.lossweights.lambda_s1
        self.loss_intra_s2, self.correct_order_intra_s2 = self.lossfunction(z_s2d, z_s2dd) 
        self.loss_intra_s2 *= self.config.lossweights.lambda_s2

        self.optimizer.zero_grad()

        self.loss_inter.backward(retain_graph=True)
        self.loss_intra_s1.backward(retain_graph=True)
        self.loss_intra_s2.backward()

        self.optimizer.step()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/loss_inter", self.loss_inter, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/loss_intra_s1", self.loss_intra_s1, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/loss_intra_s2", self.loss_intra_s2, global_step=self.globalstep)

        self.TB_writer.add_scalar("train/correct_order_inter", self.correct_order_inter, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/correct_order_intra_s1", self.correct_order_intra_s1, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/correct_order_intra_s2", self.correct_order_intra_s2, global_step=self.globalstep)

        return None
        
    def _validate(self):
        
        list_of_val_losses_inter = []
        list_of_val_losses_intra_S1 = []
        list_of_val_losses_intra_S2 = []

        with torch.no_grad():
            
            pbar_val = tqdm(total=len(self.val_data_loader), desc=f"EPOCH: {self.current_epoch}",leave=False)
            pbar_val.set_description("Validation")

            for batch_idx, batch in enumerate(self.val_data_loader):

                s1 = batch["s1"]
                s1d = batch["s1d"]
                s1dd = batch["s1dd"]

                s2 = batch["s2"]
                s2d = batch["s2d"]
                s2dd = batch["s2dd"]
                
                if self.cuda:

                    s1 = s1.cuda()
                    s1d = s1d.cuda()
                    s1dd = s1dd.cuda()

                    s2 = s2.cuda()
                    s2d = s2d.cuda()
                    s2dd = s2dd.cuda()

                # the model returns:
                # h_s1, h_s2, z_s1_inter, z_s2_inter, z_s1_intra, z_s2_intra
                h_s1, h_s2, z_s1, z_s2, _ , _ = self.model(s1, s2)
                _, _, _, _, z_s1d , z_s2d = self.model(s1d, s2d)
                _, _, _, _, z_s1dd , z_s2dd = self.model(s1dd, s2dd)
                
                loss_inter = self.lossfunction(z_s1, z_s2) * self.config.lossweights.lambda_inter
                loss_intra_s1 = self.lossfunction(z_s1d, z_s1dd) * self.config.lossweights.lambda_s1
                loss_intra_s2 = self.lossfunction(z_s2d, z_s2dd) * self.config.lossweights.lambda_s2

                list_of_val_losses_inter.append(loss_inter)
                list_of_val_losses_intra_S1.append(loss_intra_s1)
                list_of_val_losses_intra_S2.append(loss_intra_s2)

                pbar_val.update()
                
            pbar_val.close()   

            # write to tensorboard
            mean_list_of_val_losses_inter = torch.mean(torch.Tensor(list_of_val_losses_inter))
            mean_list_of_val_losses_intra_S1 = torch.mean(torch.Tensor(list_of_val_losses_intra_S1))
            mean_list_of_val_losses_intra_S2 = torch.mean(torch.Tensor(list_of_val_losses_intra_S2))

            if not self.config.validation_every_N_sampels == -1:
                self.TB_writer.add_scalar(f"val/loss_inter", mean_list_of_val_losses_inter, global_step=self.globalstep)
                self.TB_writer.add_scalar(f"val/loss_intra_S1", mean_list_of_val_losses_intra_S1, global_step=self.globalstep)
                self.TB_writer.add_scalar(f"val/loss_intra_S2", mean_list_of_val_losses_intra_S2, global_step=self.globalstep)
            
            if not self.config.validate_after_every_n_epoch == -1:
                self.TB_writer.add_scalar(f"val/epoch_loss_inter", mean_list_of_val_losses_inter, global_step=self.current_epoch)
                self.TB_writer.add_scalar(f"val/epoch_loss_intra_S1", mean_list_of_val_losses_intra_S1, global_step=self.current_epoch)
                self.TB_writer.add_scalar(f"val/epoch_loss_intra_S2", mean_list_of_val_losses_intra_S2, global_step=self.current_epoch)

            sum_all_losses = np.mean([mean_list_of_val_losses_inter,
                                      mean_list_of_val_losses_intra_S1,
                                      mean_list_of_val_losses_intra_S2])

            if sum_all_losses < self.best_metric_saveCrit:
                self.best_metric_saveCrit = mean_list_of_val_losses_inter
                self._save_checkpoint()

        return None