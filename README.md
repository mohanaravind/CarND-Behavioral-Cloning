# **Behavioral Cloning** 

![Center lane driving](/examples/auto-run.gif)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* [model.h5](https://www.dropbox.com/s/tam6gh99g42dnl9/model.zip?dl=0) containing a trained convolution neural network 
* README.md summarizing the results (what you are currently reading)

**Note**: the model.h5 file is zipped and is not part of this repository. Download it from [here](https://www.dropbox.com/s/tam6gh99g42dnl9/model.zip?dl=0) 
Why not here? 
It is bigger than 100 MB and github doesn't allow that

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
To design the model architecture for this problem I went with the [nVIDIA End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) for inspiration.
The problem in hand matches exactly with what this architecture solves. Using Keras functional model API I defined this model. This uses the following sequence of layers:

1. Input layer
```python
inp = Input(hp.input_shape, name="input")
```	 
2. Cropping layer (Trimming the top and bottom of the camera image and avoiding distraction)
```python
out = Cropping2D(cropping=hp.crop_area, name="cropping")(inp)
```	 
3. Lambda layer to scale down (resize) using tensorflow. This reduces the memory footprint.
```python
def resize_image(x, shape, scale):
    from keras.backend import tf as ktf
    dim = (shape[1] // scale, shape[0] // scale)
    return ktf.image.resize_images(x, dim)

    # Scaling
    out = Lambda(resize_image, arguments={'shape':hp.input_shape, 'scale':hp.scale}, name='scaling')(out)
```	
4. Normalization 
```python
out = Lambda(lambda x: x/127.5 - 1.0, name="normalization")(out)
```	 
5. Convolution (24 filters of 5x5) using ReLu activation
```python
out = Convolution2D(24, 5,5, activation='relu', name="convo1")(out)
```	 
6. Convolution (36 filters of 5x5) using ReLu activation
```python
out = Convolution2D(36, 5, 5, activation='relu', name="convo2")(out)
```	 
7. Convolution (48 filters of 5x5) using ReLu activation
```python
out = Convolution2D(48, 5, 5, activation='relu', name="convo3")(out)
```	 
8. Convolution (64 filters of 3x3) using ReLu activation
```python
out = Convolution2D(64, 3, 3, activation='relu', name="convo4")(out)
```	 
9. Convolution (64 filters of 3x3) using ReLu activation
```python
out = Convolution2D(64, 3, 3, activation='relu', name="convo4")(out)
```	 
10. Flatten layer
```python
out = Flatten(name="flatten")(out)
```	 
11. Fully connected layer (100 neurons)
```python
out = Dense(100, name="fully1")(out)
```	 
12. Fully connected layer (50 neurons)
```python
out = Dense(50, name="fully2")(out)
```	 
13. Fully connected layer (1 neuron outputting the steering angle)
```python
out = Dense(1, name="fully3")(out)
```	 

#### 2. Attempts to reduce overfitting in the model

The model did not show signs of overfitting. It might be overfit to this track but the learning did not show any need for a dropout layer or any other regularization method

#### 3. Model parameter tuning

Adam optimizer is used, so the learning rate was not tuned manually (model.py line 153).
Mean squared error is used to compute the loss of the model.
```python
model.compile(loss='mse', optimizer='adam')
```	 

#### 4. Appropriate training data
Training data was chosen with the target to keep the vehicle driving on the road without causing any possible hazard if a real human is a passenger. Multiple laps of center lane driving and reverse track driving was used. Also focused on repeating regions of the track where the network struggle to drive without breaking out of the road. 

For more details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I initially started by building the nVIDIA network architecture without any changes. The immediate problem I faced was I couldn't manage to run the model even once. The memory required was too much and my gpu couldn't handle it. Then I tried to reduce the number of convolution layers and fully-connected layers and managed to run.

The strategy was to try and run the car autonomously with a known model before going on to collect more data. I was using the default training data that was provided to build my initial model. By reducing the amount of training data I was able to quickly iterate over the model and iron out any coding errors. This also helped me gauge what was physically possible to run in my machine.

The approach I took is to train the network with the default data and overfit the model before trying to improve on data or tune any hyperparameter.

###### Lesson learnt:
- Lambda function needs to be pure. Cannot pass an object and consume its properties. Fails with JSON serialization
- Using lambda function inside Lambda layer fails

###### Initial attempt
Without any data augmentation and default data: The car stays on road for few meters and then goes off-road


#### 2. Final Model Architecture

The final model was an improvisation of nVIDIA network as described above

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Curves](/examples/arch.png)

#### 3. Creation of the Training Set & Training Process

This was the hardest part of this project. With the same default data training with the same network model sometime gives results that makes the car drive few yards before getting off-road and at times totally going off-road. Used both a joystick controller and a keyboard to do my training.

![Center lane driving](/examples/train.gif)

##### Training
Initially I did multiple laps of center lane driving in both the directions
![Center lane driving](/examples/center.png)

Then I did multiple training over the curves and corners
![Curves](/examples/curve.png)

Model struggled initially when it was encountering a different terrain and there was no clear demarcation of the road. So I focused to build more training data on this particular patch of the track in both the direction
![Terrain](/examples/terrain.png)


Augmenting the data by flipping the centre-camera image through its vertical axis gave a huge boost to the performance. This gave me the confidence on the importance of training data
```python
center_image_flip = np.fliplr(center_image)
center_steer_flp = - center_steer
```

##### Data augmentation
I used the left and right cameras of the car as well. Using a steering correction factor of 0.4 and again using the flipping technique the final samples grew by 6 folds from the recorded data. This gave around 90,000 training samples. At every possible point I ensured shuffling of the data set to avoid any bias

![Multiple cameras](/examples/multiple-cameras.png)

![Flipping](/examples/flipl.png)

##### Normalization & Scaling
Normalizing the data to have zero mean and equal variance is important for good training. I did this as part of my network model. Cropping the image greatly reduces the amount of pixel inputs that gets feed into the network. Also resizing the image greatly allows the model to be practically trainable in a short duration. I reduced the size of the image by scaling it down by a factor of **4**

![Cropped](/examples/cropped.jpg)

##### Process and Parameters

I found that the training did not require multiple epochs to converge. Ended up using only **2 epochs**.
Choose a batch size of **64** since that was limit of my gpu. 
![Flipping](/examples/fig.png)


### References
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

