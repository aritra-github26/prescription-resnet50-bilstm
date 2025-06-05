from pathlib import Path

import torch

from itertools import groupby
import numpy as np
from torch.autograd import Variable
from torch import nn
from torchvision.models import resnet34, resnet50  # Changed from resnet50 to resnet34
import torch.nn.functional as F
import math
from data import preproc as pp


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OCR(nn.Module):

    def __init__(self, vocab_len, hidden_dim):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a BiLSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)

        # prediction heads with length of vocab
        self.vocab = nn.Linear(hidden_dim * 2, vocab_len)  # Adjusted for bidirectional output

        # spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.query_pos = PositionalEncoding(hidden_dim, .2)

    def get_feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)   
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, inputs):  # Remove trg parameter
        """
        Forward pass through the model.
        
        Args:
            inputs: Input images (batch of images).
        
        Returns:
            Output predictions for the sequences.
        """

        # Propagate inputs through ResNet-50
        x = self.get_feature(inputs)

        # Convert from 2048 to hidden_dim feature planes
        h = self.conv(x)  # shape: (batch, hidden_dim, H, W)

        # Add spatial positional encodings
        h_shape = h.shape
        H, W = h_shape[2], h_shape[3]
        # Expand row and col embeddings to match spatial dimensions
        row_emb = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)  # (H, W, hidden_dim//2)
        col_emb = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)  # (H, W, hidden_dim//2)
        pos_emb = torch.cat([row_emb, col_emb], dim=-1).permute(2, 0, 1).unsqueeze(0)  # (1, hidden_dim, H, W)
        pos_emb = pos_emb.to(h.device)
        h = h + pos_emb  # Add positional encoding to feature map

        # Prepare input for LSTM: flatten spatial dims and permute to (batch, seq_len, feature)
        h = h.flatten(2).permute(0, 2, 1)  # (batch, seq_len, feature)

        # Add positional encoding to LSTM input features
        h = h.permute(1, 0, 2)  # (seq_len, batch, feature)
        h = self.query_pos(h)
        h = h.permute(1, 0, 2)  # (batch, seq_len, feature)

        # Pass through BiLSTM
        h, _ = self.lstm(h)  # output shape: (batch, seq_len, hidden_dim*2)

        # Permute output to (seq_len, batch, feature)
        h = h.permute(1, 0, 2)

        # Calculate output
        output = self.vocab(h)  # (seq_len, batch, vocab_size)

        return output


def make_model(vocab_len, hidden_dim=256):
    
    return OCR(vocab_len, hidden_dim)



## Old code for reference

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=128):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


# class OCR(nn.Module):

#     def __init__(self, vocab_len, hidden_dim):
#         super().__init__()

#         # create ResNet-50 backbone
#         self.backbone = resnet50()  # Omitted old backbone
#         # self.backbone = resnet34()  # Changed to ResNet34 backbone
#         del self.backbone.fc

#         # create conversion layer
#         self.conv = nn.Conv2d(2048, hidden_dim, 1)  # Omitted old conv layer for 2048 channels
#         # self.conv = nn.Conv2d(512, hidden_dim, 1)  # Adjusted conv layer for ResNet34 output channels

#         # create a BiLSTM layer
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)

#         # prediction heads with length of vocab
#         self.vocab = nn.Linear(hidden_dim * 2, vocab_len)  # Adjusted for bidirectional output

#         # output positional encodings (object queries)
#         self.decoder = nn.Embedding(vocab_len, hidden_dim)
#         self.query_pos = PositionalEncoding(hidden_dim, .2)

#         # spatial positional encodings
#         self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.trg_mask = None

#     def generate_square_subsequent_mask(self, sz):
#         mask = torch.triu(torch.ones(sz, sz), 1)
#         mask = mask.masked_fill(mask==1, float('-inf'))
#         return mask

#     def get_feature(self, x):
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)   
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)
#         return x

#     def make_len_mask(self, inp):
#         return (inp == 0).transpose(0, 1)

#     def forward(self, inputs, trg):  # Process input images and target sequences
#         """
#         Forward pass through the model.
        
#         Args:
#             inputs: Input images (batch of images).
#             trg: Target sequences (ground truth labels).
        
#         Returns:
#             Output predictions for the target sequences.
#         """

#         # Propagate inputs through ResNet-50
#         x = self.get_feature(inputs)

#         # Convert from 2048 to hidden_dim feature planes
#         h = self.conv(x)

#         # Prepare input for LSTM: flatten spatial dims and permute to (batch, seq_len, feature)
#         h = h.flatten(2).permute(0, 2, 1)  # (batch, seq_len, feature)

#         # Pass through BiLSTM
#         h, _ = self.lstm(h)  # output shape: (batch, seq_len, hidden_dim*2)

#         # Permute output to (seq_len, batch, feature)
#         h = h.permute(1, 0, 2)

#         # Getting positional encoding for target
#         trg = self.decoder(trg)
#         trg = self.query_pos(trg)

#         # Calculate output
#         output = self.vocab(h)  # (seq_len, batch, vocab_len)

#         return output


# def make_model(vocab_len, hidden_dim=256):
    
#     return OCR(vocab_len, hidden_dim)
