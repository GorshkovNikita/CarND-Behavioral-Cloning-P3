# **Behavioral Cloning** 

## Writeup


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane_driving.jpg "Center lane driving"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/recovery_driving_left_1.jpg "Recovery Image"
[image4]: ./examples/recovery_driving_left_2.jpg "Recovery Image"
[image5]: ./examples/recovery_driving_left_3.jpg "Recovery Image"
[image6]: ./examples/source.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3
filter sizes and depths between 24 and 64 (model.py lines 82-90)

The model includes RELU layers to introduce nonlinearity,
and the data is normalized in the model using a Keras lambda layer (code line 81).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 83, 87, 92, 96).

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and
right sides of the road and driving in opposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try to use one of existing architectures which was created for similar tasks.

My first step was to use a convolution neural network model similar to the NVIDIA architecture from article End to End Learning for Self-Driving Cars.
I thought this model might be appropriate because it was used for the same task as in this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set.
This implied that the model was overfitting.

To combat the overfitting, I modified the model so that in used several Dropout layers.
Then I trained the model again with these modifications.

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, 
I trained the network again with additional data for these concrete spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network
with the following layers and layer sizes:

Layer (type)               | Output Shape     |     Param #   
|--------------------------|------------------|------------------|
input_1 (InputLayer)       | (None, 160, 320, 3)  |     0        | 
cropping2d_1 (Cropping2D)  | (None, 65, 320, 3)   |     0        |
lambda_1 (Lambda)          | (None, 65, 320, 3)   |     0        | 
conv2d_1 (Conv2D)          | (None, 31, 158, 24)  |     1824     | 
dropout_1 (Dropout)        | (None, 31, 158, 24)  |     0        | 
conv2d_2 (Conv2D)          | (None, 14, 77, 36)   |     21636    | 
conv2d_3 (Conv2D)          | (None, 5, 37, 48)    |     43248    | 
dropout_2 (Dropout)        | (None, 5, 37, 48)    |     0        | 
conv2d_4 (Conv2D)          | (None, 3, 35, 64)    |     27712    | 
conv2d_5 (Conv2D)          | (None, 1, 33, 64)    |     36928    | 
flatten_1 (Flatten)        | (None, 2112)         |     0        | 
dropout_3 (Dropout)        | (None, 2112)         |     0        | 
dense_1 (Dense)            | (None, 1164)         |     2459532  | 
dense_2 (Dense)            | (None, 100)          |     116500   | 
dropout_4 (Dropout)        | (None, 100)          |     0        | 
dense_3 (Dense)            | (None, 50)           |     5050     | 
dense_4 (Dense)            | (None, 10)           |     510      | 
dense_5 (Dense)            | (None, 1)            |     11       | 

Total params: 2,712,951

Trainable params: 2,712,951

Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one in each direction (clockwise and counterclockwise) 
using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle 
would learn to return on the center of the road after steering to the side of the road.
These images show what a recovery looks like starting from the left side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Also I collected some additional data on the hardest turns of the road.

To augment the data sat, I also flipped images and angles thinking that
this would eliminate left steering bias because of counterclockwise lap.
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 18108 number of data points. 
I then preprocessed this data by modifying images so that they have zero-mean.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Notes
I wasn't able to train my model in carnd-term1 environment on my GPU. I had to create new environment 
by running the command `conda create --name tf_gpu tensorflow-gpu`. I addded the file `environment.txt` 
with information about this environment to show versions of tensorflow, cudatoolkit and other libraries 
with which my model is running properly.