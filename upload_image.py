import torch
from facenet_utils import clsMultiScaleDataLoaderCustom
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
from config import *
from model import main
from model.facenet import clsTFBaseModel, clsMultiScalePairNetsSVM
from facenet_pytorch import InceptionResnetV1


def saveFeatures(p_data_loader, p_id, tensor_feature):
    emp_feature = {'emp_code': p_id}
    cuda = torch.cuda.is_available()
    for batch_idx, batch_X in enumerate(p_data_loader):
        selfie_set = batch_X['selfie']
        if cuda:
            for k, v in selfie_set.items():
                torch.save(v.cuda(), os.path.join(feature_storage, p_id + '_{}.tf'.format(k)))
                emp_feature.update({k: v.cuda()})
        else:
            for k, v in selfie_set.items():
                torch.save(v, os.path.join(feature_storage, p_id + '_{}.tf'.format(k)))
                emp_feature.update({k: v})
    tensor_feature.append(emp_feature)


def lookUpParrentFeature(tensor_feature, emp_code, feature_id, feature):
    for x in range(0, len(tensor_feature)):
        if tensor_feature[x].get('emp_code') == emp_code:
            tensor_feature[x][feature_id] = feature
            return tensor_feature
    new_emp_code = {'emp_code': emp_code,
                    feature_id: feature}
    tensor_feature.append(new_emp_code)
    return tensor_feature


def loadFeatures(p_user_id=''):
    list_features_files = [f for f in os.listdir(feature_storage)]
    # if os.path.isfile(os.path.join(feature_storage, f)) and f.split('_')[0] == p_user_id]
    tensor_features = []
    for f in list_features_files:
        path_file = os.path.join(feature_storage, f)
        # with open(os.path.join(feature_storage, f), 'rb') as file:
        #     buffer = io.BytesIO(file.read())
        emp_code = f.split('_')[0] + '_' + f.split('_')[1]
        feature_id = f.split('_')[2].split('.')[0]
        lookUpParrentFeature(tensor_features, emp_code, feature_id, torch.load(path_file))
        # tensor_features[f.split('_')[1].split('.')[0]] = torch.load(path_file)
    return tensor_features


def extract_feature(doc, id_file, user_id, tensor_feature):
    v_img_root = "dataset"
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
    valid = clsMultiScaleDataLoaderCustom(selfie=doc, img_root=v_img_root, transform=transform_valid)
    data_loader_valid = DataLoader(valid, batch_size=params['batch_size'], shuffle=params['shuffle'],
                                   num_workers=params['num_workers'], pin_memory=True, )
    saveFeatures(data_loader_valid, user_id + '_' + id_file, tensor_feature)


tensor_features = loadFeatures()
from datetime import datetime

now = datetime.now()
basemodel_doc = InceptionResnetV1(pretrained='vggface2')

basemodel_selfie = InceptionResnetV1(pretrained='vggface2')

basenet_doc = clsTFBaseModel(basemodel_doc, num_removed_layers=0, freezed_layers=-1)

basenet_selfie = clsTFBaseModel(basemodel_selfie, num_removed_layers=0, freezed_layers=-1)

model = clsMultiScalePairNetsSVM(basenet_selfie, basenet_doc)  # clsMultiScaleNet(basenet)
v_model_file = "/Users/bao.tran/Downloads/3/facematch_svm_epoch_15_0.9575792247416961_2020_01_31_12_57_53.pth"
checkpoint = torch.load(v_model_file, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['state_dict'])
model.eval()
now = datetime.now()
print(main.compare_face(selfie='/Users/bao.tran/Downloads/3/Image99.png', tensor_feature=tensor_features, model=model))
print(datetime.now() - now)
