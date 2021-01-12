# -*- coding: utf-8 -*-
import torch.nn as nn

from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch


class clsFaceNet(nn.Module):
    def __init__(self):
        super(clsFaceNet, self).__init__()
        model_type = 'vggface2'
        self.model = InceptionResnetV1(pretrained=model_type)

    def forward(self, x):
        x = self.model(x)

        return x


class clsMultiScaleNet(nn.Module):
    def __init__(self, basemodel):
        super(clsMultiScaleNet, self).__init__()

        self.basemodel = basemodel
        self.dist = nn.PairwiseDistance(p=2, keepdim=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, doc_x1, doc_x2, doc_x3, selfie_x1, selfie_x2, selfie_x3):
        feat_doc_x1 = self.basemodel(doc_x1)
        feat_doc_x2 = self.basemodel(doc_x2)
        feat_doc_x3 = self.basemodel(doc_x3)

        feat_selfie_x1 = self.basemodel(selfie_x1)
        feat_selfie_x2 = self.basemodel(selfie_x2)
        feat_selfie_x3 = self.basemodel(selfie_x3)

        dist_x1 = self.dist(feat_doc_x1, feat_selfie_x1)
        dist_x2 = self.dist(feat_doc_x2, feat_selfie_x2)
        dist_x3 = self.dist(feat_doc_x3, feat_selfie_x3)
        return dist_x1, dist_x2, dist_x3


class clsMultiScaleNetSVM(nn.Module):
    def __init__(self, basemodel):
        super(clsMultiScaleNetSVM, self).__init__()

        self.basemodel = basemodel
        # self.cosin = nn.CosineSimilarity()
        self.dist = nn.PairwiseDistance(p=2, keepdim=True)
        self.fc = nn.Linear(in_features=3, out_features=1, bias=True)

    def forward(self, doc_x1, doc_x2, doc_x3, selfie_x1, selfie_x2, selfie_x3):
        feat_doc_x1 = self.basemodel(doc_x1)
        feat_doc_x2 = self.basemodel(doc_x2)
        feat_doc_x3 = self.basemodel(doc_x3)

        feat_selfie_x1 = self.basemodel(selfie_x1)
        feat_selfie_x2 = self.basemodel(selfie_x2)
        feat_selfie_x3 = self.basemodel(selfie_x3)

        dist_x1 = self.dist(feat_doc_x1, feat_selfie_x1)
        dist_x2 = self.dist(feat_doc_x2, feat_selfie_x2)
        dist_x3 = self.dist(feat_doc_x3, feat_selfie_x3)
        dist = torch.cat((dist_x1, dist_x2, dist_x3), dim=1)

        output = self.fc(dist)  # torch.sigmoid(self.fc(dist))

        dict_dist = {'dist_x1': dist_x1, 'dist_x2': dist_x2, 'dist_x3': dist_x3}

        return output, dict_dist


class clsMultiScaleDocIdNetSVM(nn.Module):
    def __init__(self, basemodel_selfie, basemodel_doc):
        super(clsMultiScaleDocIdNetSVM, self).__init__()

        self.basemodel_selfie = basemodel_selfie
        self.basemodel_doc = basemodel_doc
        self.dist = nn.PairwiseDistance(p=2, keepdim=True)
        self.fc3 = nn.Linear(in_features=3, out_features=1, bias=True)
        self.fc2 = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, doc_x1, doc_x2, doc_x3, doc_x0, selfie_x1, selfie_x2, selfie_x3, selfie_x0):
        feat_doc_x2 = self.basemodel_doc(doc_x2)
        feat_doc_x3 = self.basemodel_doc(doc_x3)
        feat_selfie_x3 = self.basemodel_selfie(selfie_x0)
        dist_x2 = self.dist(feat_doc_x2, feat_selfie_x3)

        dist_x3 = self.dist(feat_doc_x3, feat_selfie_x3)
        dist = torch.cat((dist_x2, dist_x3), dim=1)

        output = self.fc2(dist)
        dict_dist = {'dist_x1': 0, 'dist_x2': dist_x2, 'dist_x3': dist_x3, 'fc': output}
        return output, dict_dist


class clsMultiScalePairNetsSVM(nn.Module):
    def __init__(self, basemodel_selfie, basemodel_doc):
        super(clsMultiScalePairNetsSVM, self).__init__()

        self.basemodel_selfie = basemodel_selfie
        self.basemodel_doc = basemodel_doc
        self.dist = nn.PairwiseDistance(p=2, keepdim=True)
        self.fc3 = nn.Linear(in_features=3, out_features=1, bias=True)
        self.fc2 = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, doc_x1,doc_x2, doc_x3, doc_x0, selfie_x1,selfie_x2, selfie_x3,  selfie_x0):
        feat_doc_x1 = self.basemodel_doc(doc_x1)
        feat_doc_x0 = self.basemodel_doc(doc_x0)

        feat_selfie_x1 = self.basemodel_selfie(selfie_x1)
        feat_selfie_x3 = self.basemodel_selfie(selfie_x0)

        dist_x1 = self.dist(feat_doc_x1, feat_selfie_x3)
        dist_x2 = self.dist(feat_doc_x0, feat_selfie_x3)
        dist_x3 = self.dist(feat_doc_x1, feat_selfie_x1)

        dist = torch.cat((dist_x1, dist_x2, dist_x3), dim=1)

        output = self.fc3(dist)
        dict_dist = {'dist_x1': 0, 'fc': output}
        return output, dict_dist


class clsTFBaseModel(nn.Module):
    def __init__(self, basemodel, num_removed_layers=1, freezed_layers=None):
        super(clsTFBaseModel, self).__init__()

        if num_removed_layers > 0:
            self.model = nn.Sequential(*list(basemodel.children())[:-num_removed_layers])
        else:
            self.model = basemodel

        self.num_layers = self.fn_count_layers()

        self.num_removed_layers = num_removed_layers

        if freezed_layers is None or freezed_layers > self.num_layers:
            freezed_layers = self.num_layers // 2

        # print('AAAAAAA',freezed_layers)
        # print('BBBBBBB',self.num_layers)

        self.fn_freeze(freezed_layers)

    def fn_freeze(self, freezed_layers):

        # transfer_layer_num > 0: transfer weights from some layers begining from 1st layers
        if freezed_layers > 0:
            i_layer = 0
            for layer in self.model.children():
                i_layer += 1
                if i_layer < freezed_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True

                        # transfer_layer_num == -1: retrain the full model on a new data
        elif freezed_layers == -1:
            for param in self.model.parameters():
                param.requires_grad = True


        elif freezed_layers == 0:
            for param in self.model.parameters():
                param.requires_grad = False

                # generate a number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def fn_count_layers(self):
        n = sum(1 for layer in self.model.children())
        return n

    def forward(self, in_x):
        x = self.model(in_x)
        return x