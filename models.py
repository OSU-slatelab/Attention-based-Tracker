import pdb
import time
import numpy as np
import random
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from tqdm import tqdm
from util import *
from encoders import *

def get_mask(lens, device):
    #return (torch.arange(max(lens), device=device).expand(len(lens), max(lens)) >= torch.tensor(lens, device=device).unsqueeze(1)).float()
    mask = torch.ones(len(lens), max(lens), device=device)
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

class MulAttention(nn.Module):
    def __init__(self, input_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super(MulAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout)

        #self.linear1 = nn.Linear(input_dim, dim_feedforward)
        #self.linear2 = nn.Linear(dim_feedforward, input_dim)

        #self.norm1 = nn.LayerNorm(input_dim)
        #self.norm2 = nn.LayerNorm(input_dim)

        #self.dropout = nn.Dropout(dropout) 

    def forward(self, Q, K, lens):
        mask = get_mask(lens, Q.get_device())
        Q, K = Q.permute(1,0,2), K.permute(1,0,2)
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask, average_attn_weights=True)
        ## Add and norm
        #src = Q + self.dropout(src)
        #src = self.norm1(src)
        ### MLP
        #src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        ### Add and norm
        #src = src + self.dropout(src2)
        #src = self.norm2(src)

        return src, attn

class AddAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim):
        super(AddAttention, self).__init__()
        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.mlp_out = nn.Linear(enc_dim, output_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, lens):
        '''
        Q ==> 32xSx256
        K ==> 32xTx256
        '''
        bsz = Q.size(0)
        enc_h = self.mlp_enc(K) # 32xTx256
        dec_h = self.mlp_dec(Q) # 32xSx256
        mask = length_to_mask(lens, max_len=enc_h.size(1), device=Q.device) # 32xT

        attn = self.mlp_attn(torch.tanh(dec_h.unsqueeze(2) + enc_h.unsqueeze(1))).squeeze(-1) # 32xSxTx1
        attn = attn.masked_fill(mask.unsqueeze(1) == 0, -np.inf) # 32xSxT
        attn = self.softmax(attn)

        context = torch.bmm(attn, K) # 32xSx256
        context = self.mlp_out(context).view(bsz,-1)

        return context, attn

class Tracker(nn.Module):
    def __init__(self, args):
        super(Tracker, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        # text encoder
        self.embedding = nn.Embedding(args.vocab_size, 10)
        self.tEncoder = LstmEncoder(nLayer=args.t_layer, inDim=10, hidDim=args.hidden//2, dropout0=0.01, dropout=args.dropout)

        # pointer
        if args.attn_type == 'multiplicative':
            self.attention = MulAttention(input_dim=args.hidden, nhead=1)
        elif args.attn_type == 'additive':
            self.attention = AddAttention(enc_dim=args.hidden, dec_dim=args.hidden, attn_dim=args.hidden, output_dim=args.hidden)
        else:
            raise ValueError(f"{self.attn_type} is not implemented.")

        self.sEncoder0 = LstmLayer(args.nspeech_feat, args.hidden, dropout=args.dropout, bidirectional=False)
        self.pyrLstm = pLSTM(args.hidden, args.hidden, 2, dropout=args.dropout, bidirectional=False)
        self.sEncoder1 = LstmEncoder(args.s_layer, args.hidden, args.hidden, dropout0=args.dropout, dropout=args.dropout, bidirectional=False)

    def extract_attn(self, attn, lensS, lensT):
        out = []
        for i in range(len(lensS)):
            out.append(attn[i][:lensS[i]])    
        return torch.cat(out, dim=0)

    def forward(self, speechB, textB, lensS, lensT):
        textBE = self.embedding(textB) # 32xTx10
        textEnc, _ = self.tEncoder(textBE) # 32xTx256

        speechB, _ = self.sEncoder0(speechB) # 32x4Sx256
        speechOut0, lensS = self.pyrLstm(speechB, lensS) # 32xSx256
        speechOut, _ = self.sEncoder1(speechOut0) # 32xSx256

        _, attn = self.attention(speechOut, textEnc, lensT) # 32xSxT
        return self.extract_attn(attn, lensS, lensT)
