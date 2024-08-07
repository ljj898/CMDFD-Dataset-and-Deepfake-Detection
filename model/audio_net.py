"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch



class AudioNet(nn.Module):

    """
    An audio-only speech transcription model based on the Transformer architecture.
    Architecture: A stack of 12 Transformer encoder layers,
                  first 6 form the Encoder and the last 6 form the Decoder.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
           is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses):
        super(AudioNet, self).__init__()
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.audioDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return


    def forward(self, inputBatch):
        inputBatch = inputBatch.transpose(1, 2)
        batch = self.audioConv(inputBatch)
        batch = batch.transpose(1, 2).transpose(0, 1)
        batch = self.positionalEncoding(batch)
        batch = self.audioEncoder(batch)
        batch = self.audioDecoder(batch)
        batch = batch.transpose(0, 1).transpose(1, 2)
        batch = self.outputConv(batch)
        batch = batch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.softmax(batch, dim=2)#.softmax
        return outputBatch