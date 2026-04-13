"""
Usage: python code/visualize.py --image_path {path_to_imag.jpg} --checkpoint checkpoints/sat_best.pth --data_dir data/flickr8k/processed
"""

import argparse
import pickle
import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models import Decoder


class CNNEncoder(nn.Module):
    def __init__(self, model_type="vgg"):
        super().__init__()
        if model_type == "resnet":
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-2]  # remove pooling and FC layers
            self.model = nn.Sequential(*modules)
            self.enc_dim = 2048
        else:
            vgg = models.vgg16(pretrained=True)
            modules = list(vgg.features.children())
            self.model = nn.Sequential(*modules)
            self.enc_dim = 512

    def forward(self, images):
        out = self.model(images)  # (batch, channels, 14, 14)
        out = out.permute(0, 2, 3, 1)
        out = out.view(out.size(0), -1, out.size(3))
        return out


def visualize_attention(decoder, features, tokenizer, max_length=20):
    pass
