# Deep Transfer Learning for Image Colorization


This repository contains code for a project on colorizing grayscale landscape images using deep transfer learning.


Image colorization involves taking an input grayscale (black & white) image and then producing an output colorized image (RGB) that represents the semantic colors and tones of the input. 


The process employed in involved training a neural network model to predict the colorized version of grayscale images based on their features. The autoencoder model was used and the architecture consisted of an encoder which incorporates the `VGG16` as a feature extractor, and a decoder built using convolutional layers for upsampling and image reconstruction.


Training data used can be downloaded [here](https://drive.google.com/drive/folders/1HxQgbdN-2FMPRQIDgoskYY5dA_Ekhxk_?usp=drive_link)
