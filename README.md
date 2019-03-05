### Facial Keypoint Detection in Python using CNNs with Pytorch and OpenCV

The basic idea here is to create and train a Convolutional Neural Network (CNN) that will learn to detect facial keypoints (like corner of eyes, nose, mouth contour, eyebrows, face circumference etc.) in images of faces. 

Here is an example of images that the CNN will train with (the keypoints are overlaid in pink):
![alt text](https://github.com/shubhra/facial-keypoint-detection/blob/master/images/key_pts_example.png)
In each image there is a single face along with 68 keypoints and the (x,y) coordinates of these keypoints. These face images have been extracted from the [YouTube Database of faces](https://www.cs.tau.ac.il/~wolf/ytfaces/) and preprocessed to have keypoints associated with them.

Total number of images: 5770
Channels per image: 4 (RGBA): we'll be discarding alpha
Number of images for training: 3462
Number of images for testing: 2308

#### [1. Load and Visualize Data.ipynb](https://nbviewer.jupyter.org/github/shubhra/facial-keypoint-detection/blob/master/1.%20Load%20and%20Visualize%20Data.ipynb)

The first notebook contains code for loading the data, visualizing some of this data to understand it better and setting up relevant transforms. Here, we see that the '/data/training_frames_keypoints.csv' file contains image names and corresponding facial keypoints as a 68x2 matrix (x and y coordinates for each of the 68 keypoints). Upon visualizing a few images we quickly realize that not all images are the same size and will need to standardized for training purposes.

Using torch.utils.data.Dataset and torch.utils.data.Dataloader, we create our own dataset class to get batches of images and iterate over these images to test out the transforms on them for standardization purposes. The transforms for this data are:
* ReScale - to make all images the same size
* RandomCrop - to introduce translation invariance by cropping images randomly. To augment data, one technique to apply could be flipping of images to incorporate more of translation invariance. Data augmentation exposes the training model to additional variations without the cost of collecting and annotating more data. We're not augmenting the data with cropped images this time though. Just cropping the images for getting some translation invariance.
* Normalize - to convert the color images to grayscale images where the pixel values are in the range [0,1] and the keypoints to be normalized so that they are centered around zero and are hence in a range -1, 1
* ToTensor - to convert from numpy arrays to PyTorch tensors

Note: Instead of making these transforms as functions, we make them callable classes so any required params need not be sent in everytime these are called. To make them callable classes, the __call__ method is implemented and if any params are needed then even the __init__ method has been implemented. And note that ReScale should be called before RandomCrop.

#### [2. Define the Network Architecture.ipynb](https://nbviewer.jupyter.org/github/shubhra/facial-keypoint-detection/blob/master/2.%20Define%20the%20Network%20Architecture.ipynb) and [CNN architecture](https://github.com/shubhra/facial-keypoint-detection/blob/master/models.py)

The second notebook contains code to: 
* [architect a CNN](https://github.com/shubhra/facial-keypoint-detection/blob/master/models.py) with images as input and keypoints as output
* use the transforms from the previous notebook and get the transformed dataset
* set up hyperparameters,train and tweak, as needed, the CNN on the training data 
* test how the model performs on the test data
* visualize the features that the CNN was learning to recognize
* visualize the feature maps (feature maps are just sets of filtered images; they are the images produced by applying a convolutional kernel to an input image)

The layers in the network architecture that I built look like:  
 
  - (conv1): Conv2d(1, 16, kernel_size=(7, 7), stride=(1, 1))  
  - (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  - (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))  
  - (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))  
  - (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))  
  - (conv5): Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1))  
  - (fc1): Linear(in_features=8192, out_features=1024, bias=True)  
  - (fc1_drop): Dropout(p=0.6)  
  - (fc2): Linear(in_features=1024, out_features=136, bias=True)  
 
I settled with the current architecture which has 5 convolutional layers and reduces the filter size from 7 to 5 to 3 eventually. This seems to do a decent job but I would have like to try out the one in the paper (perhaps with a GPU setup). Maxpool layer has been added between the first two convolutional layers to avoid overfitting along with a dropout layer (p=0.6) between the two fully connected layers at the end.
 
 Number of Epochs = 3  
 Loss function = I used the Mean Squared Error loss function - [MSELoss()](https://pytorch.org/docs/master/nn.html?highlight=mseloss#torch.nn.MSELoss)  
 Optimizer = [RMSProp](https://pytorch.org/docs/master/optim.html#torch.optim.RMSprop)  
 Learning rate = 0.0001  
 
For the number of epochs, I started with 1 and when I thought I had a decent network I incremented it to 3 and then to 5. But 5 epochs was no better than 3 so I stayed at 3 epochs.  
MSE loss function is suited for regression, which directly compares a predicted value and target value. I also tried L1Loss but got better results with MSE.   
The RMSProp optimizer is an adaptive learning rate method which keeps a running average of the local gradient and adapts the learning rate	by	dividing it by	the magnitude of the running gradient. I saw better results with RMSProp than Adam optimizer given my contrained epochs.   

##### Visualizing Features  
Here is a visualization of the features of a sample test image:  
![alt text](https://github.com/shubhra/facial-keypoint-detection/blob/master/images/face_filter_sample.png)
###### First Convolutional Layer sees this:
It appears that several filters try to detech edges - vertical and horizontal. The 6th one and the last one appear to be trying to blur out noise. The 4th one and the 6th one also seem to be isolating dark and bright regions.
![alt text](https://github.com/shubhra/facial-keypoint-detection/blob/master/images/face_filter_first_cnn_layer.png)
###### Last Convolutional Layer sees this:
 Interestingly, for this layer's set of filtered images, we've reached a point where the network seems to be looking at the images as a whole mostly.
![alt text](https://github.com/shubhra/facial-keypoint-detection/blob/master/images/face_filter_last_cnn_layer.png)  

#### [3. Facial Keypoint Detection, Complete Pipeline.ipynb](https://nbviewer.jupyter.org/github/shubhra/facial-keypoint-detection/blob/master/3.%20Facial%20Keypoint%20Detection%2C%20Complete%20Pipeline.ipynb)

The third notebook contains code to:
* detect all faces in an image using OpenCV's [HaarCascade classifier](https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=cascadeclassifier#cascadeclassifier), and its [detectMultiScale](https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=cascadeclassifier#cascadeclassifier-detectmultiscale)) method
* transform the detected faces to grayscale and then to a tensor
* use the model trained in the previous network to detect facial keypoints for the detected face

Here's an example image for which a face was detected, changed to grayscale, rescaled, normalized, changed to a tensor and then facial keypoints were also detected:   
  
![Original Image](https://github.com/shubhra/facial-keypoint-detection/blob/master/images/mona_lisa.jpg)  

Here's the resulting image with keypoints overlaid on top:  
  
![Image with Keypoints](https://github.com/shubhra/facial-keypoint-detection/blob/master/images/mona_lisa_keypoints.png)

As we see, the CNN model does decently well and approximates the keypoints to a good degree. It could certainly be made much better with more time and experimentation. 












