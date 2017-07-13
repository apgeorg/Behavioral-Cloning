[//]: # (Image References)

[image1]: ./images/original.png "Original Images"
[image2]: ./images/cropped.png "Cropped Images"
[image3]: ./images/resized.png "Resized Images"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Behavioral Cloning 

The goal of this project is to build, a convolution neural network in Keras that predicts steering angles from camera images. The resulting model should successfully drive around a track without leaving the road.  

### Data Set Exploration & Pre-processing

The data set provided by Udacity contains 8036 images for each camera (center, left, right) with an image shape of 160x320x3.To get an idea what the images the look like, I visualize an image of each camera.

![alt text][image1]

#### Data Pre-processing 

As a first step, the images were cropped by 60 pixels from top and 20 pixels from bottom to remove not relevant information like the sky, background and the car front.

![alt text][image2]

As next, the images were resized to 48x128x3 to reduce the complexity of the neural network. Here is an example of an resized image of each camera. 

![alt text][image3]

### Training Data

The training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

#### Creation of the Training Set 

To capture good driving behavior, I first recorded four laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn these behavior. 

To augment the data set, I also randomly flipped images and the steering angles thinking that this would remove the bias from the model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

### Model Architecture and Training Strategy

#### Model architecture

My first approach was to trying the NVDIA architecture which is an proven architecture for autonomous driving. 
These architecture was quite enough to ensuring that the vehicle could stay on the track.

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

As a last step, I normalized the image data to a range of [0,1] by using ((pixel / 255) - 0.5), so the network can treat every feature equally.

#### 3. Model parameter tuning

The model used an adam optimizer with a fix learning rate 1e-3. The batch size was set to 32 images. The weights were initialized by a glorot uniform distribution, also called Xavier uniform distribution. The network was trained for 3 epochs on a notebook.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

### Running the Model 
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
