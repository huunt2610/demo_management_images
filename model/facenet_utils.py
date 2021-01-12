# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class clsLogging():
    @staticmethod
    def writeline(filename, line):
        with open(filename, 'a+') as hFile:
            hFile.write(line + '\n')
        hFile.close()

    @staticmethod
    def writedataset(filename, dataset):
        pd.DataFrame(dataset).to_csv(filename, index=False)


class clsTripletLoss(nn.Module):
    def __init__(self, encoder):
        super(clsTripletLoss, self).__init__()
        self.encoder = encoder
        self.D = nn.PairwiseDistance(p=2)

    def forward(self, a_doc, p_selfie, n_selfie):
        feat_a = self.encoder(a_doc)
        feat_p = self.encoder(p_selfie)
        feat_n = self.encoder(n_selfie)
        dist_p = self.D(feat_a, feat_p)
        dist_n = self.D(feat_a, feat_n)
        return dist_p, dist_n, feat_a, feat_p, feat_n


class clsAMSoftmaxLoss(nn.Module):
    def __init__(self, encoder, s, m):
        super(clsAMSoftmaxLoss, self).__init__()
        self.encoder = encoder
        self.s = s
        self.m = m  #

    def forward(self, doc, selfie):
        loss = 0.0
        return loss


class clsMultiScaleDataLoader(Dataset):
    def __init__(self, datafile, img_root='dataset', transform=None):

        self.transform_set = transform
        self.datafile = datafile
        self.img_root = img_root
        if os.path.isfile(datafile):
            self.dataset = pd.read_csv(datafile, sep=',')
        else:
            print('File %s is not existed' % (datafile))
            # raise 'Error Files'

    def __getitem__(self, index):
        # doc = Image.open(self.dataset.iloc[index, 1])
        selfie = Image.open(self.dataset.iloc[index, 0])
        target = self.dataset.iloc[index, 2]

        if self.transform_set:
            transform_f0 = self.transform_set['scale_f0']
            transform_f1 = self.transform_set['scale_f1']
            transform_f2 = self.transform_set['scale_f2']
            transform_f3 = self.transform_set['scale_f3']
            # doc_X = {'x1': transform_f1(doc),
            #          'x2': transform_f2(doc),
            #          'x3': transform_f3(doc),
            #          'x0': transform_f0(doc)
            #          }
            selfie_X = {'x1': transform_f1(selfie),
                        'x2': transform_f2(selfie),
                        'x3': transform_f3(selfie),
                        'x0': transform_f0(selfie)
                        }

        local_X = {
            # 'doc': doc_X,
                   'selfie': selfie_X,
                   'target': target,
                   'ori_doc': self.dataset.iloc[index, 1],
                   'ori_selfie': self.dataset.iloc[index, 0]
                   }

        return local_X

    def __len__(self):
        return len(self.dataset)


class clsMultiScaleDataLoaderCustom(Dataset):
    def __init__(self, selfie, img_root='dataset', transform=None):
        self.transform_set = transform
        self.selfie = selfie
        self.img_root = img_root

    def __getitem__(self, index):
        selfie = Image.open(self.selfie)
        selfie_X = {}
        if self.transform_set:
            transform_f0 = self.transform_set['scale_f0']
            transform_f1 = self.transform_set['scale_f1']
            # transform_f2 = self.transform_set['scale_f2']
            # transform_f3 = self.transform_set['scale_f3']
            selfie_X = {'x1': transform_f1(selfie),
                        # 'x2': transform_f2(selfie),
                        # 'x3': transform_f3(selfie),
                        'x0': transform_f0(selfie)
                        }

        local_X = {
                   'selfie': selfie_X,
                   'ori_selfie': self.selfie
                   }
        return local_X

    def __len__(self):
        return 1
