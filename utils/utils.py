import os 
import csv 
import cv2
import numpy as np

"""
    Read csv file.
"""
def read_csv(driving_log_csv):
    with open(driving_log_csv) as csvfile:
        reader = csv.reader(csvfile)
        return [line for line in reader]

"""
    Reads an image as RGB.
"""
def get_image(imgs_path, source_path):
    filename = source_path.split('/')[-1]
    img = cv2.imread(imgs_path + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

"""
    Crops an image.
"""
def crop(image, top=0, bottom=0, left=0, right=0):
    return image[top:image.shape[0]-bottom,left:image.shape[1]-right,:]


"""
    Flips an image horizontal. 
"""
def flip_horizontal(image):
    return cv2.flip(image, 1)

"""
    Randomly change the contrast of an image.
"""
def contrast(image):
    alpha = -.2*np.random.rand()+0.5
    beta = 100.*np.random.rand()
    img = cv2.multiply(image, np.array([alpha]))
    return cv2.add(img, np.array([beta]))

"""
    Randomly rotates an image.
"""
def rotate(image):
    rows, cols = image.shape[0], image.shape[1] 
    deg = 40.0*np.random.rand()-15
    print(deg)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1.)
    return cv2.warpAffine(image, M, (cols, rows))

"""
    Randomly change the brightness of an image.
"""
def brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 
    hsv[:,:,2] = hsv[:,:,2] * (np.random.uniform() + 0.05) 
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

"""
    Generates augmented image data. 
"""
def generate_images(X, y):
    X_augmented, y_augmented = [], [] 
    for i, angle in enumerate(y):
        if np.random.choice(2):
            X_augmented.append(brightness(X[i]))
            y_augmented.append(angle)    
        if np.random.choice(2):
            X_augmented.append(flip_horizontal(X[i]))
            y_augmented.append(-angle)
    return np.asarray(X_augmented), np.asarray(y_augmented)
    
"""
    Returns the training data and the corresponding labels. 
"""
def get_train(img_path, driving_log_csv, img_size=(360, 160)):        
    images, measurements = [], []
    lines = read_csv(driving_log_csv)
    for line in lines[1:]:
        measurement = float(line[3])
        correction = 0.
        rand = np.random.choice(3)
        if rand == 0:
            correction = 0.
        elif rand == 1:
            correction = .25
        elif rand == 2:
            correction = -.25
        images.append(cv2.resize(crop(get_image(img_path, line[rand]), 60, 20), img_size))
        measurements.append(measurement+correction) 
    return np.array(images), np.array(measurements)

"""
    Creates a directory, if not exists. 
"""    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Create directory {}".format(path))
    else:
        print("Directory {} exists".format(path))
