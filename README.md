# build a simple dog breed classification model by using transfer learning tech
CNN Project [Udacity Deep Learning Nanodegree]
![alt text](/images/Head_image.jpeg "mess-labels")

if you also want to check more details about the introduction of this project, pls check the [blog post](https://medium.com/@jobhunthanz/build-a-simple-dog-breed-classification-model-by-using-transfer-learning-tech-fb600328f02c).


## project overview
this repo includes my work for the capstion project included in the data scientist nanodegree course. This project is about the prediction of dog breed and human face by using Convolutional Neural Networks (CNNs). in this project i had learned how to build a pipeline to process the real-world, user-supplied images. also by using the transfer learning tech to build a high-accuracy model quickly, and enable to predict the dog breed with around 80% test accuracy with.

## dog breeds 
in this project a dataset including 133 breeds of dog image data is used. the following image is just for a example.
![alt text](/images/dog_breeds_examples.png "mess-labels")

# packages used in this project
numpy==1.12.1  \
keras==2.0.9 \
tensorflow-gpu==1.3.0 \
scikit-learn==0.19.1 \
opencv-python==3.3.1 \
matplotlib==2.1.0 

## how to run this project
1) download this project files
2) run the .py file
  python dog_app.py
3) also you can check the details by opening the "dog_app.html" via web browser such as chrome


## Problem Introduction
based on the information from the this website. we can see that the image classification accuracy based on the ImageNet dataset had been improved so lot from 2011, which the Top-1 accuracy(#Fig1) had been improved from around 50% to 90%. but also we got an another problem which the model is also becoming bigger and bigger(#Fig2). and in some kinds of meaning, we need to take more time to train the model to achieve the high-accuracy model with more high-performance GPUs. sometimes it can not be affordable individually. So is there any method or approach that we can get a high-accuracy model, for example, high-accuracy model to classify the dog breed, with less-training-time and no need to use not such high-performance GPU like RTX 2060 or RTX 3060?

## Strategy to solve the problem
with the development of the deep learning libs, some libs, such as keras, also can provide the pre-trained models, so that developers can apply these pre-trained model into the developer’s own project with tiny modification of the model architecture an less-training-time. this technology or approach is so-called “transfer learning”.

## Metrics details
metric used in this project is about the accuracy which measures how many observations, both positive and negative, were correctly classified. 
![alt text](/images/accuracy.png "mess-labels")

## Apply the transfer learning in this project
now we just take the dog breed classification model as a example to introduce that how to apply the transfer learning tech in the actual project.
the following figure(#Fig4) is a high-level model architecture of the image classification. the input data is a image including something, and the output is the prediction results which are the texture data.
in this dog breed classification project, we will use the dog image dataset which includes 133 breeds dog image data. and below figure(#Fig5) gives some example information about the dog breeds.

here are the steps included in this project. for more details, pls check the post [here](https://medium.com/@jobhunthanz/build-a-simple-dog-breed-classification-model-by-using-transfer-learning-tech-fb600328f02c)
Step 0: Import Datasets
Step 1: Create a CNN to Classify Dog Breeds (from Scratch)
Step 2: Use a VGG16 CNN to Classify Dog Breeds (using Transfer Learning)
Step 3: Create a InceptionV3 CNN to Classify Dog Breeds (using Transfer Learning)


## Results and accuracy improvements
we built the model from scratch with around 4% test accuracy which is too bad, then we built a specific model based on a pre-trained VGG16 model by using transfer learning, and got the test accuracy with 35% which is still not good. finally we use the pre-trained InceptioonV3 model with original Top-1 accuracy of 90%, we got the 80% test accuracy which is, in some kinds of meaning , a huge improvement. here is the actual screenshot of test accracy in this project.
 ![alt text](/images/final_test_accuracy.png) 

## the following cases may can not correctly predicted
  1) the image can not contain the whole human face, just like as follows:
   ![alt text](/test_images/human_1.jpg "mess-labels")
  2) the image both include the human face and other things,just like as follows:
   ![alt text](/test_images/dog_human_1.jpeg "mess-labels")
  3) the image both include the dog and other things like cat, just like as follows:
  ![alt text](/test_images/cat_dog_1.jpeg "mess-labels")
  4) image include something similar with human face, but not exactly the real human face
  ![alt text](/test_images/human_2.png "mess-labels")

# some additional information for this project
1) Kaggle data: https://www.kaggle.com/c/dog-breed-identification/code
2) Dog Breed Classifier With PyTorch Using Transfer Learning: 
	https://levelup.gitconnected.com/dog-breed-classifier-with-pytorch-using-transfer-learning-8f15af6f9010
