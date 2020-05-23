# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./examples/1_Road_Signs_color.png "Road Signs Color"
[image2]: ./examples/2_Traindata_histogram.png "Road Signs Histogram"
[image3]: ./examples/3_Flip_horiz.png "Horizontal Flipping"
[image4]: ./examples/4_Flip_vert.png ""
[image5]: ./examples/5_Flip_diag.png ""
[image6]: ./examples/6_Flip_both.png ""
[image7]: ./examples/7_Flip_interchange.png ""
[image8]: ./examples/8_sign_rot.png ""
[image9]: ./examples/9_sign_proj.png ""
[image10]: ./examples/10_Road_Signs_enhanced.png ""
[image11]: ./examples/11_New_signs.png ""
[image12]: ./examples/12_New_Signs_result.png ""

## Pipeline
The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

## Step 1: Loading, Exploring and Visualizing the Data

In this project, the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) has been used to train a convolutional neural network. Each image is represented as a 32x32 pixel RGB image. The number of sign images is then split into Training, Validation and Testing sets as follows.

| Data        | Number of Images | 
|-------------|------------------| 
| Training    | 34799            | 
| Validation  | 4410	         |
| Testing 	  |	12630	         |

The testing data is never looked at except at the very end to calculate model performance. The data consists of 43 variety of traffic signs. An example of each sign from the training dataset has been shown below.

![alt text][image1]

The number of images for each sign is not equal in the dataset. Some signs occur more frequently than the other. Thus, precaution needs to be taken so that the model is not biased towards more frequent signs. The frequency plot is given below.

![alt text][image2]

---

## Step 2: Data Enrichment and Preprocessing


### Image Mirroring
First, the motivation is to increase the amount of data to enhance the learning. Some of the traffic signs have an axis of symmetry i.e. they are invariant to fliping them vertically, horizontally or diagonally. An example is the "Yield" sign which as a vertical axis of symmetry passing through its center. Including this flipped data will help the model capture this inherent symmetry in some of the traffic signs. Also, some pairs of signs interchange between each other on flipping like "Turn Right" becomes "Turn Left". The Symmetric properties of signs are given in the table below

|ID  | Data                                             | Line of Symmetry | 
|----|--------------------------------------------------|------------------| 
|0   |Speed limit (20km/h)|                             |
|1   |Speed limit (30km/h)                              |Horizontal
|2   |Speed limit (50km/h)                              |
|3   |Speed limit (60km/h)                              |
|4   |Speed limit (70km/h)                              |
|5   |Speed limit (80km/h)                              |Horizontal
|6   |End of speed limit (80km/h)                       |
|7   |Speed limit (100km/h)                             |
|8   |Speed limit (120km/h)                             |
|9   |No passing                                        |
|10  |No passing for vehicles over 3.5 metric tons      |
|11  |Right-of-way at the next intersection             |Vertical
|12  |Priority road                                     |Vertical, Horizontal
|13  |Yield                                             |Vertical
|14  |Stop                                              |
|15  |No vehicles                                       |Vertical, Horizontal
|16  |Vehicles over 3.5 metric tons prohibited          |
|17  |No entry                                          |Vertical, Horizontal
|18  |General caution                                   |Vertical
|19  |Dangerous curve to the left                       |Interchange 1
|20  |Dangerous curve to the right                      |Interchange 1
|21  |Double curve                                      |
|22  |Bumpy road                                        |Vertical
|23  |Slippery road                                     |
|24  |Road narrows on the right                         |
|25  |Road work                                         |
|26  |Traffic signals                                   |Vertical
|27  |Pedestrians                                       |
|28  |Children crossing                                 |
|29  |Bicycles crossing                                 |
|30  |Beware of ice/snow                                |Vertical
|31  |Wild animals crossing                             |
|32  |End of all speed and passing limits               |Diagonal
|33  |Turn right ahead                                  |Interchange 2
|34  |Turn left ahead                                   |Interchange 2
|35  |Ahead only                                        |Vertical
|36  |Go straight or right                              |Interchange 3
|37  |Go straight or left                               |Interchange 3
|38  |Keep right                                        |Interchange 4
|39  |Keep left                                         |Interchange 4
|40  |Roundabout mandatory                              |Diagonal
|41  |End of no passing                                 |
|42  |End of no passing by vehicles over 3.5 metric tons|

Using these symmeric properties, the amount of training data is increased to 56368 images. Some newally added images are shown below.

Horizontal Symmetry: "Speed Limit 80km/h"
![alt text][image3]

Vertical Symmetry: "Ahead Only"
![alt text][image4]

Diagonal Symmetry: "End of all speed and passing limit"
![alt text][image5]

Vertical and Horizontal Symmetry: "Priority Road"
![alt text][image6]

Sign Interchange: "Keep Right"
![alt text][image7]


### Image Rotation and Projection transformation
In real world, a traffic sign can be viewed from a car at a slight camera angle or rotation. To make the learning algorithm more robust to such changes, minor modifications to training data is done and added. After this step, the training data consists of 225472 traffic sign images. A sample modification is shown below.

Rotation transformation of a sign image
![alt text][image8]

Prespective transformation of a sign image
![alt text][image9]

This step concludes the enrichment of training data.

### Grayscaling and Adaptive Histogram Equalization 
The images in training data consists of very large vaiations in the amount of lighting and brightness depending upon the time of day. To improve the contrast properties of training data, techniques of Grayscaling and Adaptive Histogram Equalization are used. This results in transformation of training data into gray 32x32 pixel images. Some examples are given below.

![alt text][image10]

### Normalization
Finally, each image is normalized such that each pixel value varies in the range (-1,1). We use the expression 
$$ norm=\frac{px-128}{128}$$

---

## Step 3: Model Architecture and Testing

The final model architecture of Convolutional Neural Network used to solve the Traffic sign classification is as follows

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x36	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x36 				    |
| Flatten   	      	| Outputs  900	 	 	 	 	 	 		    |
| Fully connected		| Outputs  300       							|
| RELU					|												|
| Fully connected		| Outputs  150       							|
| RELU					|												|
| Fully connected		| Output logits  43       						|

To train the model, hyperparameters used were EPOCHS = 25, BATCH_SIZE = 256 and RATE = 0.001. AdamOptimizer was used for learning. The final model results were:
* training set accuracy of 98.7%
* validation set accuracy of 97.0%
* test set accuracy of 94.8%

The architecture of LeNet was used as the starting point of model. Training it with un-augmented data gave results around 90% accuracy. Increasing the complexity of LeNet by increasing layer nodes lead to overfitting. Thus, the decision to generate more data was taken. Using new data with original LeNet model led to underfitting. Finally, a more complex final model was used using which much better accuracy and generalization was achieved. In our model, dropout layers weren't used because their no affect on accuracy was observed.

---

## Step 4: Testing Model on New Images

Here are eight German traffic signs that I found on the web to test the model:

![alt text][image11] 

These images seem simple to classify since there are well lit and contain no background noise. They are resized into 32x32 images, preprocessed and then fed to the model. Here are the results of the prediction:

![alt text][image12] 



The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100%. Finally, we check the top five softmax probabilities of the predictions. This is to check the confidence with which our model has made these predictions. For these images, we find that the model is almost 100% confident about its predictions. This makes sense since the images seemed easy to classify.

## References
This project is submitted as partial fulfillment of the Udacity's Self-Driving Car Engineer Nanodegree program. The helper code is available [here](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project).