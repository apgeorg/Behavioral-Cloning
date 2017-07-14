from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import utils
import numpy as np

# Seed 
seed = 42
np.random.seed(seed)

# Image dimensions
img_height, img_width = 64, 128
img_shape = (img_height, img_width, 3)


"""
    Creates a model. 
"""
def create_model(input_shape=(160, 320, 3)):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. -.5, input_shape=input_shape))
    # Convolutional 
    model.add(Conv2D(24, (5, 5), strides=(2,2), activation ='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation ='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation ='relu')) 
    model.add(Conv2D(64, (3, 3), activation ='relu'))
    model.add(Conv2D(64, (3, 3), activation ='relu'))
    # Flatten
    model.add(Flatten())
    # Fully-Connection
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])   
    return model 

"""
    Runs the training. 
"""
def train(X, y, epochs=3, batch_size=32, modelname='model.h5'):
    # Split the training set to train and validation
    trX, teX, trY, teY = train_test_split(X, y, test_size=0.2, random_state=seed) 
    # Creates the model with the disered shape
    model = create_model(img_shape)
    # Fits the model 
    model.fit(trX, trY, epochs=epochs, batch_size=batch_size, validation_data=(teX, teY), shuffle=True)
    model.save('models/' + modelname)
    print ("Model saved.")
    return model

"""
    Main 
"""
def main():
    
    # Get train data - four laps 
    X1, y1 = utils.get_train('data/track1-4laps/IMG/', 'data/track1-4laps/driving_log.csv', img_size=(img_width, img_height))  
    # Get train data -  reverse laps
    X2, y2 = utils.get_train('data/track1-2laps-rev/IMG/', 'data/track1-2laps-rev/driving_log.csv', img_size=(img_width, img_height))
    # Merge data
    X_train = np.concatenate((X1, X2), axis=0)
    y_train = np.concatenate((y1, y2), axis=0)
    # Data Augmetation 
    X_aug, y_aug = utils.generate_images(X_train, y_train)
    # Merge data
    X_train = np.concatenate((X_train, X_aug), axis=0)
    y_train = np.concatenate((y_train, y_aug), axis=0)
    
    # Start training
    model = train(X_train, y_train, epochs=3, batch_size=32, modelname="model.h5")
    
if __name__ == "__main__":
    main()
