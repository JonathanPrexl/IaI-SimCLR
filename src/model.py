from sys import implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50

class ResNetEncoder(nn.Module):

    def __init__(self, backbone, in_channels):
        super(ResNetEncoder, self).__init__()

        if backbone == "resnet18":
            model = resnet18
        elif backbone == "resnet34":
            model = resnet34
        elif backbone == "resnet50":
            model = resnet50
        else:
            raise NotImplementedError("only resnet18, 34 and 50 implemented")

        self.f = []
        for name, module in model().named_children():

            if name == 'conv1':
                # # for cifar dataset
                # # adapt first layer according to paper
                # if firstlayeradapt:
                #     module = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
                # else:
                module = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        features = torch.flatten(self.f(x), start_dim=1)
        return features


class IaISimCLR(nn.Module):
    def __init__(self, backbone_s1,
                       backbone_s2,
                       in_channel_s1,
                       in_channel_s2,
                       dim_latent_space,
                       intra_projection_active,
                       normfeatures):
        super(IaISimCLR, self).__init__()

        self.intra_projection_active = intra_projection_active
        self.normfeatures = normfeatures

        assert backbone_s1 == backbone_s2, "Not implement yet"

        self.s1_encoder = ResNetEncoder(backbone=backbone_s1, in_channels=in_channel_s1)
        self.s2_encoder = ResNetEncoder(backbone=backbone_s2, in_channels=in_channel_s2)

        if backbone_s2 == "resnet18":


            self.ph_inter_s1 = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                             nn.ReLU(inplace=True), nn.Linear(512, dim_latent_space, bias=True)) 
            
            self.ph_inter_s2 = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                             nn.ReLU(inplace=True), nn.Linear(512, dim_latent_space, bias=True)) 
            

            if self.intra_projection_active:

                self.ph_intra_s1 = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                                 nn.ReLU(inplace=True), nn.Linear(512, dim_latent_space, bias=True)) 
                
                self.ph_intra_s2 = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                                 nn.ReLU(inplace=True), nn.Linear(512, dim_latent_space, bias=True)) 
               
        else:
            raise NotImplementedError("only resnet18, 34 and 50 implemented")



    def forward(self, s1, s2):
        
        h_s1 = torch.flatten(self.s1_encoder(s1), start_dim=1)
        h_s2 = torch.flatten(self.s2_encoder(s2), start_dim=1)

        z_s1_inter = self.ph_inter_s1(h_s1)
        z_s2_inter = self.ph_inter_s2(h_s2)

        if self.intra_projection_active:
            z_s1_intra = self.ph_intra_s1(h_s1)
            z_s2_intra = self.ph_intra_s2(h_s2)

        if self.normfeatures:

            h_s1 = F.normalize(h_s1, dim=-1)
            h_s2 = F.normalize(h_s2, dim=-1)

            z_s1_inter = F.normalize(z_s1_inter, dim=-1)
            z_s2_inter = F.normalize(z_s2_inter, dim=-1)

            if self.intra_projection_active:
                z_s1_intra = F.normalize(z_s1_intra, dim=-1)
                z_s2_intra = F.normalize(z_s2_intra, dim=-1)

        if self.intra_projection_active:
            return h_s1, h_s2, z_s1_inter, z_s2_inter, z_s1_intra, z_s2_intra

        else:
            return h_s1, h_s2, z_s1_inter, z_s2_inter

