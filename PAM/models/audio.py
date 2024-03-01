import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from .htsat import HTSATWrapper

def get_audio_encoder(name: str):
    if name == "HTSAT":
        return HTSATWrapper
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))