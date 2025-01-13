from models import *
from util import *
from data import *
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.augment import SpecAugment
from torchaudio.functional import rnnt_loss
import torch.distributed as dist
import numpy as np
import copy
import pdb
import random
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)

class PointerLoss(nn.Module):
    def __init__(self, args, loss='ce'):
        super(PointerLoss, self).__init__()
        self.args = args
        self.type = loss

    def forward(self, matP, matT):
        smooth_param = 1e-10
        matT = matT.float()
        if self.type == 'ce':
            return -1. * (matT * torch.log(matP+smooth_param)).sum(dim=1).mean()
        elif self.type == 'kl':
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            return kl_loss(torch.log(matP+smooth_param), matT)
        else:
            raise TypeError(f'{self.type} loss is not implemented')

def sharpen(a, T=1.):
    a_ = a**(1./T)
    a_ = a_ / a_.sum(dim=1, keepdim=True)
    return a_

class Trainer(object):
    def __init__(self, args, data, device, optimizer, normalizer, sampler=None, rank=0, checkpoint=None):
        self.args = args
        if sampler is not None or args.test or args.get_attn:
            shuffle = False
        else:
            shuffle = True
        self.sampler = sampler
        self.rank = rank
        self.data = data
        self.device = device
        self.normalizer = normalizer
        self.optimizer = optimizer

        eff_bsz = args.batch_size / args.world_size
        self.update_after = math.ceil(eff_bsz / args.bsz_small)
        collator = Collator(args)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.bsz_small, shuffle=shuffle, num_workers=8, collate_fn=collator, pin_memory=True, sampler=sampler)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, epochs=args.nepochs, steps_per_epoch=math.ceil(1. * len(self.loader) / self.update_after), anneal_strategy='cos', pct_start=0.3)
        if checkpoint is not None and args.load_sch:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.ceLoss = PointerLoss(args, loss='ce')
        self.klLoss = PointerLoss(args, loss='kl')

    def attn_cache(self, model):
        model.eval()
        for speechB, textB, logitLens, lmax, targetLens, hTL, sTL, keyL in tqdm(self.loader):
            if speechB is None:
                continue
            lmax = speechB.size(1)
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            speechB, textB = load2gpu(speechB, self.device), load2gpu(textB, self.device)
            with torch.no_grad():
                attn = model(speechB, textB, logitLens, targetLens)
            attn_trim = attn#[0]
            attn_heads = attn_trim.cpu().numpy()
            fn = keyL[0]
            with open(f'{self.args.att_path}/{fn}.npy', 'wb') as f:
                np.save(f, attn_heads)

    def change_res(self, lst, resolution=120):
        win = resolution//40
        res = []
        for i in range(0, len(lst), win):
            seg = lst[i:i+win]
            res.append(max(seg, key=seg.count))
        return res

    def evaluate(self, model):
        model.eval()
        y_pred = []
        y_true = []
        cache = []
        for speechB, textB, logitLens, lmax, targetLens, C2WL, pointersL, keyL in tqdm(self.loader):
            if speechB is None:
                continue
            lmax = speechB.size(1)
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            speechB, textB = load2gpu(speechB, self.device), load2gpu(textB, self.device)
            C2W = C2WL[0]
            key = keyL[0]
            pointers = pointersL[0]
            with torch.no_grad():
                attn = model(speechB, textB, logitLens, targetLens)
            assert attn.size(1) == len(C2W)
            ##
            #trim attn from behind
            attn = attn[:len(pointers),:]
            attn = sharpen(attn, T=self.args.temp)
            ##
            pred_discrete = []
            for frameDist in attn:
                max_weight = 0
                word_weights = {}
                for i in range(len(frameDist)):
                    word_weights[C2W[i]] = word_weights.get(C2W[i], 0) + frameDist[i].item()
                    if word_weights[C2W[i]] > max_weight:
                        word = C2W[i]
                        max_weight = word_weights[C2W[i]]
                pred_discrete.append(word)
            #if max(pointers) != max(pred_discrete):
            #    print(f'max ptr mismatch, skipping {key}')
            #    continue
            if len(pointers) != len(pred_discrete):
                print(f'ptr len mismatch, skipping {key}')
                continue
            if self.args.res > 40:
                pointers = self.change_res(pointers, resolution=self.args.res)
                pred_discrete = self.change_res(pred_discrete, resolution=self.args.res)
            if self.args.cache_results:
                score = f1_score(pointers, pred_discrete, average='macro')
                cache.append((key, score))
            y_true.extend(pointers)
            y_pred.extend(pred_discrete)
        micro = f1_score(y_true, y_pred, average='micro')
        macro = f1_score(y_true, y_pred, average='macro')
        print(f'| Micro F1 = {micro} | Macro F1 = {macro} |')
        if self.args.cache_results:
            cache_new = sorted(cache, key=lambda x : x[1])
            with open('results_las.txt', 'w') as f:
                for k, s in cache_new:
                    f.write(f'{k}   {s}\n')

    def asr(self, model, logger):
        ##
        loss_ce_list = []
        loss_kl_list = []
        ##
        for epoch in range(self.args.epochs_done+1, self.args.nepochs+1):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            print(f'Running epoch {epoch}.')
            step = 0
            ##
            loss_ce_rec = 0.
            loss_kl_rec = 0.
            ##
            self.optimizer.zero_grad()
            for speechB, textB, logitLens, lmax, targetLens, hTL, sTL, keyL in tqdm(self.loader):
                if speechB is None:
                    continue
                model.train()
                step += 1
                lmax = speechB.size(1)
                lens_norm = [1.*(x/lmax) for x in logitLens]
                speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=epoch-1) # mean-var normalize
                if self.args.da_noise:
                    speechB = inject_seqn(speechB) # sequence noise injection
                    speechB = SpecDel(speechB, logitLens) # specaug --> del+ddel
                speechB, textB, hTL, sTL = load2gpu(speechB, self.device), load2gpu(textB, self.device), load2gpu(hTL, self.device), load2gpu(sTL, self.device)
                if self.args.ddp:
                    with model.no_sync():
                        ptrDist = model(speechB, textB, logitLens, targetLens)
                        loss_ce = self.ceLoss(ptrDist, hTL) / self.update_after
                        loss_kl = self.klLoss(ptrDist, sTL) / self.update_after
                        loss = self.args.lam * loss_ce + (1. - self.args.lam) * loss_kl
                        if torch.isnan(loss):
                           logger.info(f'skipping batch no. {step} in epoch {epoch} due to NaN loss') 
                           continue
                        loss.backward()
                        ##
                        loss_ce_rec += loss_ce.detach()
                        loss_kl_rec += loss_kl.detach()
                        ##
                else:
                    ptrDist = model(speechB, textB, logitLens, targetLens)
                    loss_ce = self.ceLoss(ptrDist, hTL) / self.update_after
                    loss_kl = self.klLoss(ptrDist, sTL) / self.update_after
                    loss = self.args.lam * loss_ce + (1. - self.args.lam) * loss_kl
                    if torch.isnan(loss):
                       logger.info(f'skipping batch no. {step} in epoch {epoch} due to NaN loss') 
                       continue
                    loss.backward()
                    ##
                    loss_ce_rec += loss_ce.detach()
                    loss_kl_rec += loss_kl.detach()
                    ##
                if step % self.update_after == 0 or step == len(self.loader):
                    nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                    self.optimizer.step() 
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if self.args.ddp:
                        dist.all_reduce(loss_ce_rec)
                        dist.all_reduce(loss_kl_rec)
                        loss_ce_list.append(loss_ce_rec.item() / dist.get_world_size())
                        loss_kl_list.append(loss_kl_rec.item() / dist.get_world_size())
                    else:
                        loss_ce_list.append(loss_ce_rec.item())
                        loss_kl_list.append(loss_kl_rec.item())
                    ##
                    loss_ce_rec = 0.
                    loss_kl_rec = 0.
                    ##
                if step % 256 == 0 and self.args.corpus == 'librispeech':
                    print(f'| loss_ce = {np.mean(loss_ce_list)} | loss_kl = {np.mean(loss_kl_list)} | lr = {self.scheduler.get_last_lr()} |')
            if self.rank==0 or not self.args.ddp:
                log = f'| epoch = {epoch} | loss_ce = {np.mean(loss_ce_list)} | loss_kl = {np.mean(loss_kl_list)} | lr = {self.scheduler.get_last_lr()} |'
                print(log)
                logger.info(log)
            ##
            loss_ce_list = []
            loss_kl_list = []
            ##
            if epoch % self.args.checkpoint_after == 0 and (self.rank==0 or not self.args.ddp):
                checkpoint = {'state_dict':model.state_dict(), 'normalizer':self.normalizer, 'optimizer':self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict(), 'epochs_done':epoch}
                save_checkpoint(checkpoint, f'{self.args.save_path}')
