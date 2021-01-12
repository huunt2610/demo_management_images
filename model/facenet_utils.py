# -*- coding: utf-8 -*-

import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class clsMultiScaleDataLoaderCustom(Dataset):
    def __init__(self, selfie, doc='', img_root='dataset', transform=None):
        self.transform_set = transform
        self.selfie = selfie
        self.doc = doc
        self.img_root = img_root

    def __getitem__(self, index):
        doc = Image.open(self.doc)
        selfie = Image.open(self.selfie)

        transform_f0 = self.transform_set['scale_f0']
        transform_f1 = self.transform_set['scale_f1']
        transform_f2 = self.transform_set['scale_f2']
        transform_f3 = self.transform_set['scale_f3']
        doc_X = {'x1': transform_f1(doc),
                 'x2': transform_f2(doc),
                 'x3': transform_f3(doc),
                 'x0': transform_f0(doc)
                 }
        selfie_X = {'x1': transform_f1(selfie),
                    'x2': transform_f2(selfie),
                    'x3': transform_f3(selfie),
                    'x0': transform_f0(selfie)
                    }

        local_X = {
            'doc': doc_X,
            'selfie': selfie_X,
            'ori_doc': self.doc,
            'ori_selfie': self.selfie
        }
        return local_X

    def __len__(self):
        return 1
