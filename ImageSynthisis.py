import torch.nn as nn
import torch.nn.functional as F
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, equal=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvTrans2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x + residual
        return x


import os
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dimension, spatial_channel):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        
        self.encoder_channels = [self.spatial_channel, 56, 112, 224, 448]
        self.encoder = nn.ModuleList()
        for i in range(1, len(self.encoder_channels)):
            self.encoder.append(Conv2D(self.encoder_channels[i-1], self.encoder_channels[i]))
        
        self.resnet = nn.ModuleList()
        self.n_resnet = 9
        for i in range(self.n_resnet):
            self.resnet.append(ResNet(self.encoder_channels[-1]))
        
        self.decoder_channels = [self.encoder_channels[-1], 224, 112, 56, 3]
        self.decoder = nn.ModuleList()
        for i in range(1, len(self.decoder_channels)):
            self.decoder.append(ConvTrans2D(self.decoder_channels[i-1], self.decoder_channels[i]))
        
    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        x = torch.sigmoid(x)
        return x

        
class Discriminator(nn.Module):
    def __init__(self, dimension, spatial_channel, avgpool):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        self.avgpool = avgpool
        
        self.pool = nn.ModuleList()
        for i in range(avgpool):
            self.pool.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
            
        self.dis_channels = [self.spatial_channel + 3, 64, 128, 256, 512]
        self.dis = nn.ModuleList()
        for i in range(1, len(self.dis_channels)):
            self.dis.append(nn.Conv2d(self.dis_channels[i-1], self.dis_channels[i], kernel_size=4))
            
    def forward(self, x):
        device = "cuda"  # get the device of the input tensor
        for i in range(len(self.pool)):
            x = self.pool[i](x)
        for i in range(len(self.dis)):
            x = self.dis[i](x)
        x = torch.sigmoid(x)
        return x
    
class GanModule(nn.Module):
    
    def __init__(self, generator=True, discriminator=True):
        super().__init__()
        self.G = None
        self.D1 = None
        self.D2 = None
        self.D3 = None
        
        self.dimension = 512
        self.spatial_channel = 32
        
        if generator:
            self.G = Generator(self.dimension, self.spatial_channel)
        
        if discriminator:
            self.D1 = Discriminator(self.dimension, self.spatial_channel, avgpool=0)
            self.D2 = Discriminator(self.dimension, self.spatial_channel, avgpool=1)
            self.D3 = Discriminator(self.dimension, self.spatial_channel, avgpool=2)
            self.label_real = 1
            self.label_fake = 0
            
    def forward(self, x):
        return self.generate(x)
    
    def generate(self, spatial_map):
        assert spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension), f'[Image Synthesis : generate] Expected input spatial_map shape {(-1, self.spatial_channel, self.dimension, self.dimension)}, but received {spatial_map.shape}.'
        photo = self.G(spatial_map)
        assert photo.shape[1:] == (3, self.dimension, self.dimension), f'[Image Synthesis : generate] Expected output shape {(1, 3, self.dimension, self.dimension)}, but yield {photo.shape}.'
        return photo
    
    def discriminate(self, spatial_map, photo):
        assert spatial_map.shape[0] == photo.shape[0], f'[Image Synthesis : discriminate] Input spatial_map has {spatial_map.shape[0]} batch(es), but photo has {photo.shape[0]} batch(es).'
        assert spatial_map.shape[1:] == (self.spatial_channel, self.dimension, self.dimension), f'[Image Synthesis : discriminate] Expected input spatial_map shape {(-1, self.spatial_channel, self.dimension, self.dimension)}, but received {spatial_map.shape}.'
        assert photo.shape[1:] == (3, self.dimension, self.dimension), f'[Image Synthesis : discriminate] Expected input photo shape {(-1, 3, self.dimension, self.dimension)}, but received {photo.dimension}.'
        
        spatial_map_photo = torch.cat((spatial_map, photo), 1)
        patch_D1 = self.D1(spatial_map_photo)
        patch_D2 = self.D2(spatial_map_photo)
        patch_D3 = self.D3(spatial_map_photo)
        return patch_D1, patch_D2, patch_D3
    
    



import torch

class BCE:
    def __init__(self):
        self.criterion = torch.nn.BCELoss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)
    
class MSE:
    def __init__(self):
        self.criterion = torch.nn.MSELoss()
    
    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)


class L1:
    def __init__(self):
        self.criterion = torch.nn.L1Loss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)

import torch
from torchvision import transforms
import gc
torch.cuda.empty_cache()
gc.collect()


class Perceptual:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    perceptual_layer = ['4', '9', '14', '19']

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.model.to(self.device)
        self.criterion = L1()
    
    def compute(self, prediction, ground_truth):
        prediction = self.preprocess(prediction)
        ground_truth = self.preprocess(ground_truth)
        
        loss = 0
        for layer, module in self.model.features._modules.items():
            prediction = module(prediction)
            ground_truth = module(ground_truth)
            if layer in self.perceptual_layer:
                loss += self.criterion.compute(prediction, ground_truth)
        return loss