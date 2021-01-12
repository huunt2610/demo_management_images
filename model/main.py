from facenet_pytorch import MTCNN
import os
import sys

sys.path.append(os.getcwd())

from PIL import Image
from imutils.face_utils import FaceAligner, rect_to_bb
import imutils
import cv2
import dlib

import torch

from model.facenet import clsTFBaseModel, clsMultiScalePairNetsSVM
from model.facenet_utils import clsMultiScaleDataLoaderCustom
from facenet_pytorch import InceptionResnetV1

from torch.autograd import Variable
from torchvision import transforms as tfs
from torch.utils.data import DataLoader

import torch.optim as optim

import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/bao.tran/Downloads/3/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)
mtcnn = MTCNN()


def face_alignment(input_file, output_face_file, is_selfie):
    bResult = 1
    tmp = 'tmp_image.jpg'
    try:

        if is_selfie:
            # Face alignment
            image = cv2.imread(input_file)
            image = imutils.resize(image, width=800, height=800)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for rect in rects:
                # extract the ROI of the *original* face, then align the faces
                # using facial landmarks
                try:
                    (x, y, w, h) = rect_to_bb(rect)
                    # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)

                    faceAligned = fa.align(image, gray, rect)
                    # print(faceAligned)
                    # im = Image.fromarray(np.uint8(faceAligned)*255)
                    cv2.imwrite(tmp, faceAligned)

                    img = Image.open(tmp)
                    # detect again
                    _, _ = mtcnn(img, save_path=output_face_file, return_prob=True)

                    # print("SAVE")
                except:
                    bResult = 0
                    print("Cannot save the aligned image from selfie")
                    continue
        else:
            img = Image.open(input_file)
            # detect again
            _, _ = mtcnn(img, save_path=output_face_file, return_prob=True)

    except:
        bResult = 0
        print("Cannot save the aligned image (%s) from selfie (%d) (%s)" % (output_face_file, is_selfie, input_file))

    return bResult


def compare_face(selfie, tensor_feature, threshold=0.4):
    v_img_root = "dataset"
    v_model_file = "/Users/bao.tran/Downloads/3/facematch_svm_epoch_15_0.9575792247416961_2020_01_31_12_57_53.pth"

    params = {'lr': 0.001,
              'momentum': 0.9,
              'batch_size': 1,
              'n_output_class': 1,
              'shuffle': True,
              'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              'max_epoches': 10,
              'weight_decay': 0.0001,
              'num_workers': 0,
              'num_devices': 2
              }

    # define data pipeline
    # 1.1 define transformation

    mean_norm = [0.485, 0.456, 0.406]
    std_norm = [0.229, 0.224, 0.225]

    resolution0 = 128
    resolution = [80, 100, 128]

    transform_valid = {
        'scale_f0': tfs.Compose(
            [tfs.Resize(size=resolution0),
             tfs.ToTensor(),  # normalized to [0,1]
             tfs.Normalize(mean=mean_norm, std=std_norm)  # Imagenet standards
             ]
        ),
        'scale_f1': tfs.Compose(
            [tfs.Resize(size=resolution[0]),
             tfs.ToTensor(),  # normalized to [0,1]
             tfs.Normalize(mean=mean_norm, std=std_norm)  # Imagenet standards
             ]
        ),
        'scale_f2': tfs.Compose(
            [tfs.Resize(size=resolution[1]),
             tfs.ToTensor(),  # normalized to [0,1]
             tfs.Normalize(mean=mean_norm, std=std_norm)  # Imagenet standards
             ]
        ),
        'scale_f3': tfs.Compose(
            [tfs.Resize(size=resolution[2]),
             tfs.ToTensor(),  # normalized to [0,1]
             tfs.Normalize(mean=mean_norm, std=std_norm)  # Imagenet standards
             ]
        )
    }

    # 1.2 define dataset

    valid = clsMultiScaleDataLoaderCustom(selfie, img_root=v_img_root, transform=transform_valid)

    # 1.3 define data load

    data_loader_valid = DataLoader(valid, batch_size=params['batch_size'], shuffle=params['shuffle'],
                                   num_workers=params['num_workers'], pin_memory=True, )

    basemodel_doc = InceptionResnetV1(pretrained='vggface2')

    basemodel_selfie = InceptionResnetV1(pretrained='vggface2')

    basenet_doc = clsTFBaseModel(basemodel_doc, num_removed_layers=0, freezed_layers=-1)

    basenet_selfie = clsTFBaseModel(basemodel_selfie, num_removed_layers=0, freezed_layers=-1)

    model = clsMultiScalePairNetsSVM(basenet_selfie, basenet_doc)  # clsMultiScaleNet(basenet)

    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'],
                          weight_decay=params['weight_decay'])
    checkpoint = torch.load(v_model_file, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for batch_idx, batch_X in enumerate(data_loader_valid):
        # doc_set = batch_X['doc']
        selfie_set = batch_X['selfie']

        # doc_x1, doc_x2, doc_x3, doc_x0 = doc_set['x1'], doc_set['x2'], doc_set['x3'], doc_set['x0']
        selfie_x1, selfie_x2, selfie_x3, selfie_x0 = selfie_set['x0'], selfie_set['x1'], selfie_set['x1'], \
                                                     selfie_set['x0']

        optimizer.zero_grad()
        for feature in tensor_feature:
            with torch.no_grad():
                doc_x1, doc_x2, doc_x3, doc_x0 = Variable(feature['x0']), Variable(feature['x1']), \
                                                 Variable(feature['x1']), Variable(feature['x0'])
                selfie_x1, selfie_x2, selfie_x3, selfie_x0 = Variable(selfie_x1), Variable(selfie_x2), Variable(
                    selfie_x3), Variable(selfie_x0)
                output, dict_dist = model(doc_x1, doc_x2, doc_x3, doc_x0, selfie_x1, selfie_x2, selfie_x3, selfie_x0)

            output = output.detach().cpu().numpy().squeeze()

            preds = np.where(output <= threshold, 0, 1)
            if output <= threshold:
                return output, feature['emp_code']
