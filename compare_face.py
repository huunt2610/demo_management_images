from config import *
from os import listdir
from os.path import isfile, join
import json


def getDistance(f1, f2):
    if f1 == f2:
        dt = 0.01
    else:
        dt = 1
    return dt


def compareOperator(d1, d2):
    return d1 > d2


class FaceUtils:
    feature_warehouse = []
    threshold = 0.01

    def __init__(self):
        self.feature_warehouse = []

    def loadFeatureLib(self):
        self.feature_warehouse = [f for f in listdir(feature_storage)
                                  if isfile(join(feature_storage, f)) and f.split('.')[-1] == 'json']
        return 1

    def compareFace(self, input_feature_img):
        distance = 1
        _id = 0
        for f in self.feature_warehouse:
            id_tmp = f.split('.')[-2]
            with open(os.path.join(feature_storage, f)) as fi:
                data = json.load(fi)
                result = getDistance(data, input_feature_img)
                if compareOperator(distance, result):
                    distance = result
                    _id = id_tmp
        return _id, distance


fu = FaceUtils()
if fu.loadFeatureLib():
    print(fu.feature_warehouse)

image_test = json.load(open(os.path.join(feature_storage, 'U002.json')))
emp_code, rs_distance = fu.compareFace(image_test)
print(emp_code, rs_distance)





