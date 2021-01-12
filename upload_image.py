import torch
from facenet_utils import clsMultiScaleDataLoaderCustom
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
from config import *


def saveFeatures(p_data_loader, p_id):
    cuda = torch.cuda.is_available()
    for batch_idx, batch_X in enumerate(p_data_loader):
        selfie_set = batch_X['selfie']
        if cuda:
            for k, v in selfie_set.items():
                torch.save(v.cuda(), os.path.join(feature_storage, p_id + '_{}.tf'.format(k)))
        else:
            for k, v in selfie_set.items():
                torch.save(v, os.path.join(feature_storage, p_id + '_{}.tf'.format(k)))


def loadFeatures(p_user_id=''):
    list_features_files = [f for f in os.listdir(feature_storage)]
    # if os.path.isfile(os.path.join(feature_storage, f)) and f.split('_')[0] == p_user_id]
    tensor_features = []
    for f in list_features_files:
        path_file = os.path.join(feature_storage, f)
        # with open(os.path.join(feature_storage, f), 'rb') as file:
        #     buffer = io.BytesIO(file.read())
        emp_code = f.split('_')[0]
        emp_feature = {
            'emp_code': emp_code,
            'emp_feature': torch.load(path_file)
        }
        tensor_features.append(emp_feature)
        # tensor_features[f.split('_')[1].split('.')[0]] = torch.load(path_file)
    return tensor_features


def extract_feature(doc, id_file, user_id):
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
    saveFeatures(data_loader_valid, user_id + '_' + id_file)

