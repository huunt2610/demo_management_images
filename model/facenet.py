# -*- coding: utf-8 -*-
import torch.nn as nn

from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch


class clsFaceNet(nn.Module):
    # model_type= ['vggface2', 'casia-webface' ]
    #
    def __init__(self):
        super(clsFaceNet, self).__init__()
        model_type = 'vggface2'
        self.model = InceptionResnetV1(pretrained=model_type)

    def forward(self, x):
        # print('clsFaceNet_x:',x.size())
        x = self.model(x)

        return x


class clsMultiScaleNet(nn.Module):
    def __init__(self, basemodel):
        super(clsMultiScaleNet, self).__init__()

        self.basemodel = basemodel
        # self.cosin = nn.CosineSimilarity()
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
        # print('feat_doc_x1',feat_doc_x1.size())
        # print('feat_selfie_x1',feat_selfie_x1.size())
        # print('dist_x1',dist_x1)

        dist_x2 = self.dist(feat_doc_x2, feat_selfie_x2)
        # print('dist_x2',dist_x2.size())

        dist_x3 = self.dist(feat_doc_x3, feat_selfie_x3)
        # print('dist_x3',dist_x3.size())

        # fc1 = torch.sigmoid(self.fc(dist_x1))
        # fc2 = torch.sigmoid(self.fc(dist_x2))
        # fc3 = torch.sigmoid(self.fc(dist_x3))

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

        # print('feat_selfie_x3:', feat_selfie_x3.size())

        dist_x1 = self.dist(feat_doc_x1, feat_selfie_x1)
        # print('feat_doc_x1',feat_doc_x1.size())
        # print('feat_selfie_x1',feat_selfie_x1.size())
        # print('dist_x1',dist_x1)

        dist_x2 = self.dist(feat_doc_x2, feat_selfie_x2)
        # print('dist_x2',dist_x2.size())

        dist_x3 = self.dist(feat_doc_x3, feat_selfie_x3)
        # print('dist_x3',dist_x3.size())

        # fc1 = torch.sigmoid(self.fc(dist_x1))
        # fc2 = torch.sigmoid(self.fc(dist_x2))
        # fc3 = torch.sigmoid(self.fc(dist_x3))

        # print('dist_x3:', dist_x3.size())

        dist = torch.cat((dist_x1, dist_x2, dist_x3), dim=1)

        # print('dist:', dist.size())

        # x = dist.view(-1, 96)

        # print('dist:', dist.size())
        # print('x:', x.size())

        # x = dist.view(dist_x1.size(0),3,-1)

        # print('x:', x.size())

        output = self.fc(dist)  # torch.sigmoid(self.fc(dist))

        dict_dist = {'dist_x1': dist_x1, 'dist_x2': dist_x2, 'dist_x3': dist_x3}

        return output, dict_dist


class clsMultiScaleDocIdNetSVM(nn.Module):
    def __init__(self, basemodel_selfie, basemodel_doc):
        super(clsMultiScaleDocIdNetSVM, self).__init__()

        self.basemodel_selfie = basemodel_selfie
        self.basemodel_doc = basemodel_doc
        # self.cosin = nn.CosineSimilarity()
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
