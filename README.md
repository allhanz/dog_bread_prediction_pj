# dog_bread_classifier

CNN Project [Udacity Deep Learning Nanodegree]

## project oview
this repo includes my work for the capstion project included in the data scientist nanodegree course. This project is about the prediction of dog breed and human face by using Convolutional Neural Networks (CNNs). in this project i had learned how to build a pipeline to process the real-world, user-supplied images. also by using the transfer learning tech, we no need to train the model from zero, and it becomes possible even we do not have the high performance GPU. in my own, i just use RTX 2070 (8GB) to train my own model only for 20 epoches and achieved around 80% test accuracy. 

## packages used in this project
keras==2.0.9
tensorflow-gpu==1.3.0
scikit-learn==0.19.1
numpy==1.12.1
opencv-python==3.3.1
matplotlib==2.1.0


## how to run this project
1) download this project files
2) run the .py file
  python dog_app.py
3) also you can check the details by ony opening the "dog_app.html" via web browser such as chrome

## some prediction issues
  the following cases may can not be predicted correctly, and also checked in this project.
  1) the image can not contain the whole human face, just like as follows:
   ![alt text](/test_images/human_1.png "mess-labels")
  2) the image both include the human face and other things,just like as follows:
   ![alt text](/test_images/dog_human_1.png "mess-labels")
  3) the image both include the dog and other things like cat, just like as follows:
  ![alt text](/test_images/cat_dog_1.png "mess-labels")
  4) image include something similar with human face, but not exactly the real human face
  ![alt text](/test_images/human_2.png "mess-labels")

# some additional information for this project

1) Kaggle data: https://www.kaggle.com/c/dog-breed-identification/code
2) Dog Breed Classifier With PyTorch Using Transfer Learning: 
	https://levelup.gitconnected.com/dog-breed-classifier-with-pytorch-using-transfer-learning-8f15af6f9010
