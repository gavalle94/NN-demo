<img src="cover.jpg" />

# NN-demo
A simple demo in which a Neural Network is trained to recognize hand-written digits

## Project overview
### What is the project for
The neural network is trained in order to recognize hand-written digits. It is a simple classification problem, solved trying several parameters configuration, network architectures, training algorithms and loss functions.

This demo was the first laboratory for the "Deep Learning" course we have followed at Eurecom.

### How to access the project files
You can find both the original Python Notebook code and its HTML exported version, that is more portable in terms of readability. As you can see, the dataset and all the notebook required files are provided.

## Technical details
### The architecture
A multi-layers feedforward neural network, with only one hidden layer (whose size is variable). Input layer size (784 neurons) and output layer one (10 neurons) are fixed by the problem in analysis.

### Training algorithms
The gradient descent variants are examinated and compared each other, in different contexts. We want to see what happens when changing things like the learning rate or the number of hidden neurons.

### Loss functions
Both the Mean Squared Error (MSE) and the Cross-Entropy loss functions have been used and compared.

### The PyTorch attempt
You will see that there is an attempt to convert the entire work in PyTorch code: anyway, the final result is not great. It wasan extra activity, meant as an opportunity to explore the functionalities offered by already available Python libraries.

We think there is a bug somewhere in the code and we will try to fix it in the near future.

## Credits
<a href="https://github.com/MrAngius" target="_blank">ANGIUS Marco</a> and <a href="https://github.com/gavalle94" target="_blank">AVALLE Giorgio</a> - Ⓒ2018
