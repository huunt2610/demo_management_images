from facenet_pytorch import MTCNN
import os
import sys
from PIL import Image
from imutils.face_utils import FaceAligner, rect_to_bb
import imutils
import cv2
import dlib
import torch
from torch.autograd import Variable
from torchvision import transforms as tfs
import uuid

sys.path.append(os.getcwd())
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/bao.tran/Downloads/3/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)
mtcnn = MTCNN()


def face_alignment(input_file, outputs_path):
    output_face_file = outputs_path + str(uuid.uuid4()) + '.jpg'
    try:
        img = Image.open(input_file)
        _, _ = mtcnn(img, save_path=output_face_file, return_prob=True)

    except:
        print("not detect face")
        return None
    return output_face_file


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


def compare_face(selfie, tensor_feature, model, threshold=0.6, threshhold_2=0.55):
    new_path = face_alignment(selfie, 'outputs/')
    if new_path is None:
        return 1, 'can_not_detect_face'
    image = Image.open(new_path)
    selfie_x0 = transform_valid['scale_f0'](image).unsqueeze_(0)
    selfie_x1 = transform_valid['scale_f1'](image).unsqueeze_(0)
    selfie_x2 = transform_valid['scale_f2'](image).unsqueeze_(0)
    selfie_x3 = transform_valid['scale_f3'](image).unsqueeze_(0)
    output_value = 1
    emp_code = None
    for feature in tensor_feature:
        with torch.no_grad():
            doc_x0, doc_x1, doc_x2, doc_x3 = Variable(feature['x0']), Variable(feature['x1']), Variable(
                feature['x2']), Variable(feature['x3'])
            selfie_x0, selfie_x1, selfie_x2, selfie_x3 = Variable(selfie_x0), Variable(selfie_x1), Variable(
                selfie_x2), Variable(selfie_x3)
            output, dict_dist = model(doc_x1, doc_x0, doc_x1, doc_x0, selfie_x1, selfie_x0, selfie_x1, selfie_x0)
        output = output.detach().cpu().numpy().squeeze()
        if output <= threshold:
            return output, feature['emp_code']
    return output_value, emp_code
