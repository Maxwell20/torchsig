from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.dataset import SignalDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
import numpy as np
import pickle
import lmdb
import os

classes = [
    "cw_spike",
]
num_classes = len(classes)
level = 0
include_snr = True

# Seed the dataset instantiation for reproduceability
pl.seed_everything(1234567891)

dataset = ModulationsDataset(
    classes=classes,
    use_class_idx=False,
    level=level,
    num_iq_samples=2000,
    num_samples=int(num_classes * 100),
    include_snr=include_snr,
    
)

idx = 600
if include_snr:
    data, (modulation, snr) = dataset[idx]
else:
    data, modulation = dataset[idx]

print("Dataset length: {}".format(len(dataset)))
print("Number of classes: {}".format(num_classes))
print("Data shape: {}".format(data.shape))
print("Example modulation: {}".format(modulation))
if include_snr:
    print("SNR: {}".format(snr))