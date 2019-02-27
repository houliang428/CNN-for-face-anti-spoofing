# CNN-for-face-anti-spoofing

## Description
This is my graduation project and it is based on " Deep Learning for Face Anti-Spoofing: An End-to-End Approach" published by Yasar Abbas Ur Rehamn, Lai Man Po and Mengyang Liu. Please check [paper](https://ieeexplore.ieee.org/document/8166863).    

I built three CNN models Model A,B,C based on VGG-11. And it ultimately realized accuracies of 93.74%,91.15%,92.12% for identifying fake face image.

## Configuration 
Language: Python   
Library: OpenCV    
Platform: Tensorflow     
Dataset: CASIA-FASD   

## Introduction 
The face anti-spoofing is an technique that could prevent face-spoofing attack. For example, an intruder might use a photo of the legal user to "deceive" the face recognition system. Thus it is important to use the face anti-spoofint technique to enhance the security of the system.    
The flow char of my work is as belows:
![](./images/.png)    

## Prepare Dataset
CASIA-FASD datasets are consist of videos, each of which is made of 100 to 200 video frames. For each video, I captured 30 frames (with the same interval between each frame). Then, with the [Haar_classifier](https://github.com/opencv/opencv/tree/master/data/haarcascades), I was able to crop a person’s face from an image. These images make up training datasets and test datasets.
![](./images/.png)     

## Training loss:
#### Model A:
![](./images/.png)  

#### Model B:
![](./images/.png) 

#### Model C:
![](./images/.png) 
