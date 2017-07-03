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
        images.append(cv2.imread(cur_path))
        measurements.append(float(line[3]))
    return np.array(images), np.array(measurements)
