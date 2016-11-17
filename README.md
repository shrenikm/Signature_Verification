# Signature_Verification

A signature verification system using Artificial Neural Networks.

# Data

Training data is obtained by taking pictures of signatures and using Python scripts to threshold and preprocess the image. Noise, translational and scale variation is added to each of the training image to produce more images to ensure a robust model.

# Implementation

The data is fed into a single layer neural network implementaion and the weights are trained. Other signatures can now be verified using the same procedure of preprocessing the image and feed-forwarding.
