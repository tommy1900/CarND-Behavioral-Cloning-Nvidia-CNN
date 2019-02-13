# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My model consists of a convolution neural network based on the Nvidia CNN model (model.py lines x-x) :
1. Normalization using a Keras Lambda function
2. Three 5x5 convolution layers (2x2 striding)
3. Two 3x3 convolution layers
4. Three fully-connected layers 
(All with Relu activation functions) 

Note -----> This generates the 'model.h5' file

To combat the overfitting, the model contains dropout layers in order to reduce overfitting (model.py lines x): 
1. Two dropouts (with a keep probability of 0.5) were added after the two convolution layers and the first fully-connected layer. 
2. Skipped data/image when the car is not moving
3. Reduced number of epochs based on the loss of training set and validation set

The Adam optimizer was used (default parameters) with a MSE loss function. The final output layer is a fully-connected layer with a single neuron. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I :
1. Introduced more training data by driving more tracks forward and backward
2. Introduced some "recovery situation" by intentionally run into the track line and quickly steer back.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers:
1. Normalization using a Keras Lambda function
2. Cut off top 50 pix and bottom pix with Cropping2D function 
3. 5x5 convolution layers (2x2 striding, size 24, 5, 5)
4. 5x5 convolution layers (2x2 striding, size 36, 5, 5)
5. 5x5 convolution layers (2x2 striding, size 48, 5, 5)
6. First Dropout by 0.5
7. 3x3 convolution layers (64, 3, 3)
8. 3x3 convolution layers (64, 3, 3)
9. A flatten layer
10. Fully-connected layers with dense 100
11. Second Dropout by 0.5
12. Fully-connected layers with dense 50
13. Fully-connected layers with dense 10
14. Final fully-connected layers 1 single output 

Note -----> This generates the 'model1.h5' file

Here is a visualization of the architecture:

![image01][./home/workspace/CarND-Behavioral-Cloning-P3/Project_NN_structure.png "Model Visualization"]

#### 3. Creation of the Training Set & Training Process
1. I generated my first training model solely with the center lane driving image sample data in the workspace folder, It was driving well for the first few turns, but it finally got stuck at the beginning of the "bridge". 
2. I then introduced the left and right view of the camera with 0.2 steer wheel correction. It got better but can still drive off the track if the turn curve is aggressive. 
3. I then made a new track data driving backward, which should help to reinforce the model regarding the different driving situation (more different turning situations). I was using a keyboard to play the simulator, which is not very easy to control plus the delay in the remote workspace. But It does generate a lot vehicle recovering situation from the left side and right sides of the road back to center so that the vehicle would learn how to react if the road yellow line is too close to the camera.
4. With the new data set and two epoch training, the car is able to autonomous drive on the road without running off the track.
And here is an example image of center lane driving:
![image02][./home/workspace/CarND-Behavioral-Cloning-P3/center_2016_12_01_13_30_48_287.jpg "Dataset Center Image"]

After the collection process, I had around 25,000 data points (I lose my own recorded data every time I log off the workspace). I then preprocessed this data by appending the image in one array and steering measurements in another array. I also cut off the top 50 pix and bottom 20 pix of the image, and use the rest as my region of interest. I finally randomly shuffled the dataset and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or underfitting. The ideal number of epochs was 2, I tried with 3 or 4 epochs, but the loss stopped decreasing after the 2nd epochs finished.

#------------------RESUBMIT---------------------
* I finally realized that I was still overfitting the model in the previous submit, which makes the car partially left the track at some points 
* With:
	1. Less traning data 
	2. Smaller batch size 
	3. Introduced nonlinear activation functions in fully connected layers
* I finally got a better model result (check model_test3.h5 and run2.mp4)