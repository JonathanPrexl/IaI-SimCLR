import torch
from torch.utils.data import BatchSampler
import numpy as np

class LocationSampler(BatchSampler):
    
    def __init__(self, neighbourSampels,
                       percentageRandom,
                       lenDS,
                       batch_size):
        
        self.numBatches = lenDS // batch_size
        self.batch_size = batch_size
        self.neighbourSampels = neighbourSampels
        self.numRandomSampels = int( batch_size/100 * percentageRandom)
        self.seasons = ["ROIs1158_spring","ROIs1970_fall","ROIs1868_summer","ROIs2017_winter"]

    def __iter__(self):
        
        for bidx in range(self.numBatches):

            outIndices = []
            
            # pic a random season
            # one of the four
            season = self.seasons[ np.random.randint(4) ]
            
            # for that season pick a random 
            # location... 
            all_locations = list( self.neighbourSampels[season].keys() )
            location = np.random.choice(all_locations)
            
            # pick a random set of imgs from that certain location
            num_of_potential_imgs = len( self.neighbourSampels[season][location] )
            sampels_to_pic = self.batch_size - self.numRandomSampels
            choosenOnes = np.random.permutation(self.neighbourSampels[season][location])[:sampels_to_pic]
            outIndices += list(choosenOnes)
            
            # fill up with random stuff
            for i in range(self.numRandomSampels):
                season = self.seasons[ np.random.randint(4) ]
                all_locations = list( self.neighbourSampels[season].keys() )
                location = np.random.choice(all_locations)
                num_of_potential_imgs = len( self.neighbourSampels[season][location] )
                outIndices.append( np.random.permutation(self.neighbourSampels[season][location])[0] )
            

            # for large BS if we did not hit the BS yet then fill until 
            # we reached.. some locations just have less then 512 items
            while len(outIndices) < self.batch_size:
                season = self.seasons[ np.random.randint(4) ]
                all_locations = list( self.neighbourSampels[season].keys() )
                location = np.random.choice(all_locations)
                num_of_potential_imgs = len( self.neighbourSampels[season][location] )
                outIndices.append( np.random.permutation(self.neighbourSampels[season][location])[0] )

            assert len(outIndices) == self.batch_size, f"BS {self.batch_size}, {len(outIndices)}"
            
            yield outIndices



if __name__ == "__main__":

    from torch.utils.data import Dataset, DataLoader,SequentialSampler, RandomSampler
    from omegaconf import OmegaConf
    from dataloader import Sen12MS  


    cfg_DualSimCLR = OmegaConf.load('./configs/DualSimCLR.yaml')
    dl_DualSimCLR = Sen12MS(cfg_DualSimCLR,"train")
    
    
    for batch in DataLoader(dl_DualSimCLR, batch_sampler=LocationSampler(dl_DualSimCLR.neighbourSampels,
                                                                         50,
                                                                         len(dl_DualSimCLR),
                                                                         cfg_DualSimCLR.dataloader.batch_size)):
        print("x")
        break