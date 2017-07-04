import os 
import csv 
import cv2
import numpy as np

def read_csv(driving_log_csv):
    with open(driving_log_csv) as csvfile:
        reader = csv.reader(csvfile)
        return [line for line in reader]

def get_image(imgs_path, source_path):
    filename = source_path.split('/')[-1]
    return cv2.imread(imgs_path + filename)
    
def get_train(IMG_path, driving_log_csv):        
    images = []
    measurements = []
    correction = 0.2
    lines = read_csv(driving_log_csv)
    for line in lines[1:]:
        img = get_image(IMG_path, line[0])
        img_left = get_image(IMG_path, line[1])
        img_right = get_image(IMG_path, line[2])
        measurement = float(line[3])
        images.append(img)
        measurements.append(measurement)
        images.append(cv2.flip(img, 1))
        measurements.append(-1.*measurement)
        images.append(img_left)
        measurements.append(measurement+correction)
        images.append(img_right)
        measurements.append(measurement-correction)
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
