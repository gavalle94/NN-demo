import time
import random
from math import ceil
import numpy as np
from copy import deepcopy
from utils import *
from transfer_functions import * 


class NeuralNetwork(object):
    '''
    3 layers NN: input, hidden and output
    '''
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, transfer_f=sigmoid, transfer_df=dsigmoid):
        """
        input_layer_size: number of input neurons
        hidden_layer_size: number of hidden neurons
        output_layer_size: number of output neurons
        iterations: number of iterations
        learning_rate: initial learning rate
        """

        # initialize transfer functions
        self.transfer_f = transfer_f
        self.transfer_df = transfer_df

        # initialize layer sizes
        self.input_layer_size = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden_layer_size = hidden_layer_size+1 # +1 for the bias node in the hidden layer 
        self.output_layer_size = output_layer_size

        # create randomized weights Yann LeCun method in 1988's paper ( Default values)
        self.weights_init()

    def weights_init(self,wi=None,wo=None):
        '''
        Initialize the weight matrixes
        '''
        input_range = 1.0 / self.input_layer_size ** (1/2)
        if wi is not None:
            self.W_input_to_hidden = deepcopy(wi) # weights between input and hidden layers
        else:
            self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input_layer_size, self.hidden_layer_size-1))
        if wo is not None:
            self.W_hidden_to_output = deepcopy(wo) # weights between hidden and output layers
        else:
            self.W_hidden_to_output = np.random.uniform(size = (self.hidden_layer_size, self.output_layer_size)) / np.sqrt(self.hidden_layer_size)

    def train(self, data, validation_data, iterations=300, learning_rate=5.0, batch_size=None):
        '''
        Train the NN over a dataset: loss function = MSE
        '''
        # check the batch_size value
        if(batch_size is None):
            # Gradient descent
            batch_size = len(data[0])
        if(batch_size <= 0):
            raise ValueError('batch size value is not correct')
            
        # we want to keep the time needed for the training
        start_time = time.time()
        # reset variables
        training_accuracies = []
        validation_accuracies = []
        errors = []
        # transpose the dataset: will be useful later
        dataset = np.transpose(data).tolist()
        # initialize best results variables
        best_val_acc = None
        best_i2h_W = self.W_input_to_hidden
        best_h2o_W = self.W_hidden_to_output
        # number of batches
        n_batch = int(len(data[0])/batch_size + 0.5)
        
        # iterations over the training set
        for it in range(iterations):
            # change the order of the samples
            random.shuffle(dataset)
            # retrieve features and labels
            inputs  = [x[0] for x in dataset]
            targets = [x[1] for x in dataset]
            # iterations over all the batches
            for i in range(0, len(inputs), batch_size):
                # reset NN output matrix
                self.o_output = np.ones((batch_size, self.output_layer_size))
                # define the current batch
                end = min(i+batch_size, len(inputs))
                batch_inputs = inputs[i : end]
                batch_targets = targets[i : end]
                # feedforward step
                self.feedforward(batch_inputs)
                # backpropagation step
                self.backpropagate(batch_targets, learning_rate=learning_rate)
                # compute the error on the current batch
                error = np.square(batch_targets - self.o_output)
            # compute accuracies, to be printed for the final graph
            training_accuracies.append(100*self.predict(data)/len(data[0]))
            validation_accuracies.append(100*self.predict(validation_data)/len(validation_data[0]))
            # ...better results obtained?
            if best_val_acc is None or validation_accuracies[-1] > best_val_acc:
                best_i2h_W = self.W_input_to_hidden
                best_h2o_W = self.W_hidden_to_output
                best_val_acc = validation_accuracies[-1]
                       
        # save best results as parameters of the NN
        self.W_input_to_hidden = best_i2h_W
        self.W_hidden_to_output = best_h2o_W
        
        # display obtained results
        print("Training time:", time.time()-start_time)
        plot_train_val(t=range(1, iterations+1), st=training_accuracies, sv=validation_accuracies, hn=self.hidden_layer_size-1, lr=learning_rate)
        
    def train_xe(self, data, validation_data, iterations=300, learning_rate=5.0, batch_size=None):
        '''
        Train the NN over a dataset: loss function = cross-entropy
        '''
        # check the batch_size value
        if(batch_size is None):
            # Gradient descent
            batch_size = len(data[0])
        if(batch_size <= 0):
            raise ValueError('batch size value is not correct')
            
        # we want to keep the time needed for the training
        start_time = time.time()
        # reset variables
        training_accuracies = []
        validation_accuracies = []
        n = len(data[0])
        n_batches = ceil(len(data[0])/batch_size)
        # transpose the dataset: will be useful later
        dataset = np.transpose(data).tolist()
        # initialize best results variables
        best_val_acc = None
        best_i2h_W = self.W_input_to_hidden
        best_h2o_W = self.W_hidden_to_output
        # number of batches
        n_batch = int(len(data[0])/batch_size + 0.5)
        
        # iterations over the training set
        for it in range(iterations):
            # change the order of the samples
            random.shuffle(dataset)
            # retrieve features and labels
            inputs  = [x[0] for x in dataset]
            targets = [x[1] for x in dataset]
            # we want to compute the losses for the last iteration
            error = 0.0
            xe = 0.0
            # iterations over all the batches
            for i in range(0, len(inputs), batch_size):
                # reset NN output matrix
                self.o_output = np.ones((batch_size, self.output_layer_size))
                # define the current batch
                end = min(i+batch_size, len(inputs))
                batch_inputs = inputs[i : end]
                batch_targets = targets[i : end]
                # feedforward step
                self.feedforward_xe(batch_inputs)
                # backpropagation step
                self.backpropagate_xe(batch_targets, learning_rate=learning_rate)
                # compute the error on the current batch
                xe -= np.sum(np.multiply(
                    batch_targets,
                    np.log(self.o_output)
                ))
                error += np.sum(1.0/n * np.square(batch_targets - self.o_output))
            # compute accuracies, to be printed for the final graph
            training_accuracies.append(100*self.predict_xe(data)/len(data[0]))
            validation_accuracies.append(100*self.predict_xe(validation_data)/len(validation_data[0]))
            # ...better results obtained?
            if best_val_acc is None or validation_accuracies[-1] > best_val_acc:
                best_i2h_W = self.W_input_to_hidden
                best_h2o_W = self.W_hidden_to_output
                best_val_acc = validation_accuracies[-1]
                       
        # save best results as parameters of the NN
        self.W_input_to_hidden = best_i2h_W
        self.W_hidden_to_output = best_h2o_W
        
        # display obtained results
        print("Training time:", time.time()-start_time)
        print("MSE loss:", error)
        print("XE loss:", xe) 
        plot_train_val(t=range(1, iterations+1), st=training_accuracies, sv=validation_accuracies, hn=self.hidden_layer_size-1, lr=learning_rate)

    def predict(self, test_data):
        """ 
        Evaluate performance by counting how many examples in test_data are correctly evaluated. 
        """
        # reset NN output matrix
        self.o_output = np.ones((len(test_data[0]), self.output_layer_size))
        # feedforward the data
        self.feedforward(test_data[0])
        answer = np.argmax(test_data[1], axis=1).reshape(len(test_data[0]),1)
        prediction = np.argmax(self.o_output, axis=1).reshape(len(test_data[0]), 1)
        count = len(test_data[0]) - np.count_nonzero(answer - prediction)
        return count
    
    def predict_xe(self, test_data):
        """ 
        Evaluate performance by counting how many examples in test_data are correctly evaluated. 
        """
        # reset NN output matrix
        self.o_output = np.ones((len(test_data[0]), self.output_layer_size))
        # feedforward the data
        self.feedforward_xe(test_data[0])
        answer = np.argmax(test_data[1], axis=1).reshape(len(test_data[0]),1)
        prediction = np.argmax(self.o_output, axis=1).reshape(len(test_data[0]), 1)
        count = len(test_data[0]) - np.count_nonzero(answer - prediction)
        return count

