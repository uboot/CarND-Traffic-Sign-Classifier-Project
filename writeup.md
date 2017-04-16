#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/uboot/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Summary statistics

The code for this step is contained in the second code cell of the IPython notebook.  

I calculated summary statistics of the traffic signs data set:

* The size of the training set is ?
* The size of the validation set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Exploratory visualization

The code for this step is contained in the third code cell of the IPython notebook. It contains a visualization of 36 randomly selected images of the data set. Each time the code in this cell is re-run different images are output. Note the rather large differences in image brightness of the data.

![alt text][visualization.png]

###Design and Test a Model Architecture

####1. Preprocessing

The code for this step is contained in the fourth and fifth code cells of the IPython notebook.

As a first step, I decided to convert the images to grayscale because traffic signs are designed to be identified without color information. Still the extra color information could increase the performance of the classifier it has problems to distinguish between signs of different color. Secondly, equalized the histogram of the traffic signs to compensate for the different image brightnesses.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][preprocessed_image.png]


####2. Training, validation and testing

Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the provided data for training (train.p) and validation (validation.p).
The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook. With this data I developed and optimized the model and the training the parameter such that the accuracy of the model for the validation was above 93%. After computation of the final model (lenet) I evaluated it using the provided training data (test.p).


####3. Model architecture

Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. It is basically LeNet with additional dropout layers after the fully connected layers: 

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 Grayscale image   			|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			|
| Convolution 2x2	| 2x2 stride, valid padding, outputs 10x10x16	|
| RELU			|						|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16	|
| Fully connected	| outputs 400x120    				|
| RELU			|						|
| Dropout		|						|
| Fully connected	| outputs 120x84    				|
| RELU			|						|
| Dropout		|						|
| Fully connected	| outputs 84x43    				|


####4. Training


Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth and ninth cell of the ipython notebook. 

To train the model, I used an Adam gradient descent optimizer with a learning rate of 0.0005. For a given epoch I shuffled the training data and divided it into batches of size 128. Each batch was used to run step of the gradient descent optimizer. This process was run over 100 epochs. The dropout rate during the training process was set to 0.6 for both dropout layers.

####5. Finding the solution

Because traffic signs after the preprocessing have a certain similarity to graphical data (such as the handwritten digits) I started from the LeNet architecture. Originally I chose a training rate 0.001 and 10 epochs. From there I first increased the number of epochs and observed a stagnating validation accuracy after some epochs. Next, I iteratively decreased the training rate. At some point during this process I added the dropout layers after the fully connected layers to reduce overfitting. This lead to a satisfying validation accuracy. I tried to move the dropout layers to the convolutional layers of the LeNet architecture but the results were better with the orignal approach.

Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the test accuracy of the model is located in the tenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 96.1% 
* test set accuracy of 93.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Traffic signs from the internet

Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][internet-examples/0.png] ![alt text][internet-examples/1.png] ![alt text][internet-examples/2.png] ![alt text][internet-examples/3.png] ![alt text][internet-examples/4.png]

The first two images should be rather easy to classify. The third image is a graphical depiction of a sign and does differ in proportions from a real sign. The last two images are distorted and distorted and rotated respectively. 

####2. Performance

Here are the results of the prediction:

| Image                 |     Prediction	        	| 
|:---------------------:|:-------------------------------------:| 
| No passing     	| No passing   				| 
| Yield     		| Yield 				|
| Speed limit (30km/h)	| Speed limit (30km/h)			|
| Speed limit (30km/h)	| Speed limit (30km/h)			|
| Speed limit (30km/h)	| Turn right ahead			|


The model managed to classify all but the last of the images. The performance could have been improved by training the mode with augmented (i.e. rotated) image data.

####3. Softmax probabilities

Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first two images the model is very sure (almost 100%) about the correct result.

For the third image, the model is relatively sure about the correct (speed limit 30km/h) result but assigns a signifcant value to end of speed limit (80km/h). This seems reasonable as these signs are indeed similar. In this situation it could be beneficial to use color information (red colored speed limit versus grayscale end of speed limit). These are the probabilities for this situation:

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .78         		| Speed limit (30km/h)  			| 
| .22     		| End of speed limit (80km/h)			|

The situation is very similar for the fourth image:

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .77        		| Speed limit (30km/h)  			| 
| .22     		| End of speed limit (80km/h)			|
| .01     		| Speed limit (80km/h)		        	|

The (wrongly) classifies the last image as turn right ahead. It is interesting that it is very sure about this misclassification and that the correct result is not among the top predictions:

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .98        		| Turn right ahead 		        	| 
| .02     		| Stop		                        	|

Obviously, it is absolutely necessary to augment the training data if one wants to classify rotated images as in this examples.

