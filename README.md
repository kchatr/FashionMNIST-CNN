# FashionMNIST-CNN

## Preface
Fashion MNIST is a dataset that is modeled after the MNIST dataset, which comprises of 70,000 28x28 pixel images of handwritten digits. Fashion MNIST consists of 70,000 grayscale 28x28 images of various clothing articles, taken from Zalando. There are 10 possible labels for each article. 

The objective of this model is to be able to accurately classify an unknown article of clothing in the same format. 
This was done previously through a standard densely-connected multilayer perceptron; this model employs a different network architecture, with the goal of increasing the accuracy of the classifications through using a convolutional neural network (CNN).

## Algorithm
This model uses a network with two convolutional layers with maxpooling on top of a single dense layer with 128 neurons. 
The input and first layer are 2-D convolutional layers which both use the ReLU activation function. 
The resulting convoltuional image is then flattened and passed to a dense layer with 128 neurons that uses the ReLU activation function. 
Finally, the output layer uses the Softmax activation function in order to output a probability distribution representative of the model's confidence that a label is accurate.

The model is trained over 5 epochs with a batch size of 32. It uses the Adam optimization function for gradient descent and calculates the model's loss using Sparse Categorical Cross Entropy. 

## Results
The final model has an average loss of 0.231 and an accuracy of 0.917; in comparison, the first model had an average testing loss of 0.336 and an accuracy of 0.882. Thus, by making use of convolutional layers, the model had a 3.97% increase in accuracy.  

Achieving an accuracy of over 90% on a test dataset is phenomenal, and showcases the benefit convolutional networks can provide for image related models.
