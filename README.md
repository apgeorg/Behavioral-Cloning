[//]: # (Image References)

[image1]: ./images/original.png "Original Images"
[image2]: ./images/cropped.png "Cropped Images"
[image3]: ./images/resized.png "Resized Images"
[image4]: ./images/flipped.png "Flipped Image"
[image5]: ./images/brightness.png "Changed Brightness"
[image6]: ./images/augmented.png "Augmented Images"
[image7]: ./images/model.png "Model Architecture"

## Behavioral Cloning 

The goal of this project is to build, a convolution neural network for end-to-end driving in a simulator using Tensorflow and Keras that predicts steering angles from camera images. The resulting model should successfully drive around a track without leaving the road.  

### Data Exploration & Pre-processing

The data set provided by Udacity contains 8036 images for each camera (center, left, right) with an image shape of 160x320x3.To get an idea what the images the look like, I visualize an image of each camera.

![alt text][image1]

#### Data Pre-processing 

As a first step, the images were cropped by 60 pixels from top and 20 pixels from bottom to remove not relevant information like the sky, background and the car front.

![alt text][image2]

As next, the images were resized to 64x128x3 to reduce the complexity of the neural network. Here is an example of an resized image of each camera. 

![alt text][image3]

### Training Data

The training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

#### Creation of the Training Set 

To capture good driving behavior, I first recorded four laps on track one using center lane driving with recovering gentle from left side and right sides back to center road, so that the vehicle would learn to drive centered. Further, I recorded two laps in the opposite direction on track one to have a more balanced training set. 

For the final training set I randomly select images from the left, right or center camera of the recorded data. Choosing randomly left and right images while adding a small correction of 0.25 for the left camera angle and substracting these from the right camera angle, it helps to teach the model to correct the car to the center of the track.

#### Augmentation

Data Augmentation was used because track one contains more left turns than right turns. To compensate the data set, I randomly flipped images and the steering angles. For example, here is an image that has been flipped:

![alt text][image4]

Furthermore, I also randomly adjust the brightness as an augmentation technique. 

![alt text][image5]

After performing data augmentation the total number of the training set was almost doubled. Here are some randomly augmented images from the training set: 

![alt text][image6]

Finally, I randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

### Model Architecture and Training Strategy

#### Model architecture

The overall strategy for deriving a model architecture was to trying the NVIDIA architecture [1] which is an proven architecture for autonomous driving. These architecture was good enough to ensure that the vehicle could stay on the track. 

The figure below shows the network architecture, which consists of nine layers, including a normalization layer, five convolutional layers, and three fully connected layers.

![alt text][image7]

The first layer of the network performs image normalization (pixel / 255 - 0.5) by using a Keras lambda layer. According to NVIDIA, performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.

The first three convolutional layers uses a 2×2 stride, a 5×5 filter size and depths between 24 and 48. The final two convolutional layers are non-strided convolutions with a 3×3 filter size and a depth of 64 followed by three fully connected layers.

The model includes RELU layers to introduce nonlinearity. 

#### Training & Model parameters 

The model used an adam optimizer with a fix learning rate of 1e-4. The batch size was set to 32 images. The weights were initialized by a glorot uniform distribution, also called Xavier uniform distribution. The network was trained for 3 epochs. Training with more epochs does not significantly decrease the mean squared error.  

#### Evaluation

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

### Running the Model 
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py models/model.h5
```
### References 
[1] https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
