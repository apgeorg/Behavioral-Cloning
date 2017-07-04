import os 
import csv 
import cv2
import numpy as np

def read_csv(driving_log_csv):
    with open(driving_log_csv) as csvfile:
        reader = csv.reader(csvfile)
        return [line for line in reader]


def get_train(IMG_path, driving_log_csv):        
    images = []
    measurements = []
    lines = read_csv(driving_log_csv)
    for line in lines[1:]:
        src_path = line[0]
        filename = src_path.split('/')[-1]
        cur_path =  IMG_path + filename
        img = cv2.imread(cur_path)
        measurement = float(line[3])
        images.append(img)
        images.append(cv2.flip(img, 1))
        measurements.append(measurement)
        measurements.append(measurement * -1.)
    return np.array(images), np.array(measurements)

"""
    Creates a directory, if not exists. 
    path: new path
"""    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Create directory {}".format(path))
    else:
        print("Directory {} exists".format(path))
