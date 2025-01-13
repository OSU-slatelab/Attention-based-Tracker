import torch
import random
import json
import ast
import re
import os
import time
import pandas as pd
import pdb
import string
import numpy as np
import torchaudio
import torchaudio.transforms as AT
import copy
from util import *
from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow

NPZ = ['fisher', 'swbd']

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, win_len=25, hop_length=10, n_fft=400, n_mels=80, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.compute_stft = STFT(sample_rate=sample_rate, win_length=win_len, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)
        self.sr = sample_rate

    def get_filterbanks(self, signal):
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(row['audio_file'])
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        return self.get_filterbanks(wav)

class AsrDataset(SpeechDataset):
    def __init__(self, args, csv_path, win_len=25, hop_length=10, n_fft=400, n_mels=80, sample_rate=16000, train=True):
        super(AsrDataset, self).__init__(csv_path, win_len, hop_length, n_fft, n_mels, sample_rate)
        self.args = args
        self.train = train
        if args.gt_path:
            self.gtMap = {}
            self.gtDf = pd.read_csv(args.gt_path)
            for i, row in self.gtDf.iterrows():
                self.gtMap[row['key']] = i

    def fix_path(self, path):
        if self.args.dont_fix_path:
            return path
        if self.args.corpus == 'readr':
            return path.replace('/data/corpora2/reading_races/data/', '/research/nfs_fosler_1/vishal/audio/readr/')
        elif self.args.corpus == 'librispeech':
            return path.replace('/data/corpora2/librispeech/LibriSpeech/', '/local/scratch/LibriSpeech/LibriSpeech/')
        elif self.args.corpus == 'cmu_kids':
            return path.replace('/data/data24/scratch/sunderv/', '/research/nfs_fosler_1/vishal/audio/')#fsxe/fsxe1be2.wav

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.args.corpus in ['librispeech', 'cmu_kids', 'readr']:
            wav, org_sr = torchaudio.load(self.fix_path(row['audio_file']))
        elif self.args.corpus == 'timit':
            wav, org_sr = torchaudio.load(row['audio_file'])
        else:
            raise NotImplementedError
        if org_sr != self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        if self.args.corpus == 'cmu_kids':
            key = str(row['id'])
        elif self.args.corpus == 'readr':
            key = str(row['utt_id'])[:-4]
        else:
            key = str(row['cnum'])+'-'+str(row['utt_id'])
        #align_path = os.path.join('/research/nfs_fosler_1/vishal/alignments', self.args.corpus, key+'.npz')
        align_path = os.path.join(self.args.alignment_path, self.args.corpus, key+'.npz')
        if not os.path.isfile(align_path) and not self.args.get_attn and not self.args.test:
            #print(align_path)
            return None
        hardTarget = None
        softTarget = None
        if not self.args.get_attn and not self.args.test:
            aligns = np.load(align_path)
            hardTarget = torch.from_numpy(aligns['hard'])
            softTarget = torch.from_numpy(aligns['soft'])
            if hardTarget.shape[1] != len(clean4asr(row['utterance'].replace('-', ' '))) + 2:
                #print(key)
                return None
        if self.args.gt_path:
            gtRow = self.gtDf.iloc[self.gtMap[key]]
            pointers = ast.literal_eval(gtRow['discrete'])
            C2W = charI2W(clean4asr(row['utterance'].replace('-', ' ')))
            return self.get_filterbanks(wav), clean4asr(row['utterance'].replace('-', ' ')), C2W, pointers, key
        return self.get_filterbanks(wav), clean4asr(row['utterance'].replace('-', ' ')), hardTarget, softTarget, key

class Collator(object):
    def __init__(self, args):
        self.args = args

    def pad_and_cat(self, tensors):
        if self.args.get_attn:
            return None
        if len(tensors) == 0:
            return
        max_size_dim1 = max([tensor.size(1) for tensor in tensors])
        padded_tensors = [torch.nn.functional.pad(tensor, (0, max_size_dim1 - tensor.size(1), 0, 0)) for tensor in tensors]
        return torch.cat(padded_tensors, dim=0)

    def __call__(self, lst):
        speechL = [x[0].squeeze(0) for x in lst if x and x[0].size(1) > 2 and len(x[1]) > 1]
        textL = [x[1] for x in lst if x and x[0].size(1) > 2 and len(x[1]) > 1]
        if not self.args.gt_path:
            hTL = self.pad_and_cat([x[2] for x in lst if x and x[0].size(1) > 2 and len(x[1]) > 1])
            sTL = self.pad_and_cat([x[3] for x in lst if x and x[0].size(1) > 2 and len(x[1]) > 1])
        else:
            C2WL = [x[2] for x in lst if x and x[0].size(1) > 2 and len(x[1]) > 1]
            pointersL = [x[3] for x in lst if x and x[0].size(1) > 2 and len(x[1]) > 1]

        keyL = [x[4] for x in lst if x and x[0].size(1) > 2 and len(x[1]) > 1]
        if len(speechL) == 0 or len(textL) == 0:
            return None, None, None, None, None, None, None, None

        pack1 = pack_sequence(speechL, enforce_sorted=False)
        speechB, logitLens = pad_packed_sequence(pack1, batch_first=True)
        lmax = speechB.size(1)

        textF = []
        mergeL = []
        for x in textL:
            a, b = convert_tok2id(x)
            textF.append([ASR_TOK2ID["<sos>"]]+a+[ASR_TOK2ID["<eos>"]])
            mergeL.append(b)

        textIn = [torch.tensor(x) for x in textF]

        pack2 = pack_sequence(textIn, enforce_sorted=False)
        textB, targetLens = pad_packed_sequence(pack2, batch_first=True)

        if self.args.gt_path:
            return speechB, textB, logitLens, lmax, targetLens, C2WL, pointersL, keyL

        return speechB, textB, logitLens, lmax, targetLens, hTL, sTL, keyL
