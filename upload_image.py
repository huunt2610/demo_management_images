import os
import sys
import torch
from facenet_utils import clsMultiScaleDataLoaderCustom
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
import json


# function to add to JSON
def write_json(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def extract_feature(doc, image_name, user_id):
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

    cuda = torch.cuda.is_available()
    for batch_idx, batch_X in enumerate(data_loader_valid):
        selfie_set = batch_X['selfie']
        if cuda:
            selfie_x1, selfie_x2, selfie_x3, selfie_x0 = selfie_set['x1'].cuda(), selfie_set['x2'].cuda(), selfie_set[
                'x3'].cuda(), selfie_set['x0'].cuda()

        else:
            selfie_x1, selfie_x2, selfie_x3, selfie_x0 = selfie_set['x1'], selfie_set['x2'], selfie_set['x3'], \
                                                         selfie_set[
                                                             'x0']

        arr1 = selfie_x1.detach().numpy().tolist()
        arr2 = selfie_x2.detach().numpy().tolist()
        arr3 = selfie_x3.detach().numpy().tolist()
        arr0 = selfie_x0.detach().numpy().tolist()
        file_name = f'features_folder/{user_id}.json'
        r = {image_name: [{1: arr1}, {2: arr2}, {3: arr3}, {0: arr0}]}
        r = json.dumps(r)
        json_data = json.loads(r)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        # if os.path.exists(file_name):
        #     print('2')
        #     r = {image_name: [{1: arr1}, {2: arr2}, {3: arr3}, {0: arr0}]}
        #     r = json.dumps(r)
        #     json_data = json.loads(r)
        #     with open(file_name) as json_file:
        #         data = json.load(json_file)
        #         data.append(json_data)
        #     write_json(data, filename=file_name)
        # else:
        #     r = [{image_name: [{1: arr1}, {2: arr2}, {3: arr3}, {0: arr0}]}]
        #     r = json.dumps(r)
        #     json_data = json.loads(r)
        #     with open(file_name, 'w', encoding='utf-8') as f:
        #         json.dump(json_data, f, ensure_ascii=False, indent=4)


import datetime
s = datetime.datetime.now()
print(s)
path = '/Users/huu.nguyen/Documents/SmartPay/Temp/crawling_google_images_2/1.jpg'
name = 'clinton'
id = 'U001'
extract_feature(path, name, id)
e = datetime.datetime.now()
print(e)
print(e-s)
