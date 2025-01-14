######################################
# Assignement 2 for CSC420
# MNIST clasification example
# Author: Jun Gao
######################################

import numpy as np

def cross_entropy_loss_function(prediction, label):
    # compute the cross entropy loss function between the prediction and ground truth label.
    # prediction: the output of a neural network after softmax. It can be an Nxd matrix, where N is the number of samples,
    #           and d is the number of different categories
    # label: The ground truth labels, it can be a vector with length N, and each element in this vector stores the ground truth category for each sample.
    # Note: we take the average among N different samples to get the final loss.
    log_vals = -np.log(prediction[np.arange(prediction.shape[0]), label])
    return np.mean(log_vals)

def sigmoid(x):
    # compute the softmax with the input x: y = 1 / (1 + exp(-x))
    return 1/(1+np.exp(-x))

def softmax(x):
    #  compute the softmax function with input x.
    #  Suppose x is Nxd matrix, and we do softmax across the last dimention of it.
    #  For each row of this matrix, we compute x_{j, i} = exp(x_{j, i}) / \sum_{k=1}^d exp(x_{j, k})
    numerator = np.exp(x-np.max(x, axis=1, keepdims=True))
    denominator = np.sum(numerator, axis=1, keepdims=True)
    return numerator/denominator

class OneLayerNN():
    def __init__(self, num_input_unit, num_output_unit):
        # Random Initliaize the weight matrixs for a one-layer MLP.
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        self.weights = np.random.randn(num_input_unit, num_output_unit)
        self.biases = np.zeros((1, num_output_unit))

    def forward(self, input_x):
        # Compute the output of this neural network with the given input.
        # Suppose input_x is an Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute output: z = softmax (input_x * W_1 + b_1), where W_1, b_1 are weights, biases for this layer
        # Note: If we only have one layer in the whole model and we want to use it to do classification,
        #       then we directly apply softmax **without** using sigmoid (or relu) activation
        print(input_x.shape)
        self.biased_vals = np.dot(input_x, self.weights) + self.biases
        self.final_pred = softmax(self.biased_vals)
        return self.final_pred

    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        # given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        derivative_wrt_preds = self.final_pred - np.identity(self.final_pred.shape[1])[label]
        derivative_wrt_biased_vals = self.final_pred*(1-self.final_pred) * derivative_wrt_preds
        derivative_wrt_weights = np.dot(derivative_wrt_biased_vals.T, input_x)
        derivative_wrt_biases = np.sum(derivative_wrt_biased_vals, axis=0)

        self.weights -= learning_rate * derivative_wrt_weights.T
        self.biases -= learning_rate * derivative_wrt_biases

# [Bonus points] This is not necessary for this assignment
class TwoLayerNN():
    def __init__(self, num_input_unit, num_hidden_unit, num_output_unit):
        # Random Initliaize the weight matrixs for a two-layer MLP with sigmoid activation,
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        self.l1_weights = np.random.randn(num_input_unit, num_hidden_unit)
        self.l1_biases = np.zeros((1, num_hidden_unit))

        self.l2_weights = np.random.randn(num_hidden_unit, num_output_unit)
        self.l2_biases = np.zeros((1, num_output_unit))

    def forward(self, input_x):
        # Compute the output of this neural network with the given input.
        # Suppose input_x is Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute: first layer: z = sigmoid (input_x * W_1 + b_1) # W_1, b_1 are weights, biases for the first layer
        # Compute: second layer: o = softmax (z * W_2 + b_2) # W_2, b_2 are weights, biases for the second layer
        self.l1_biased_vals = np.dot(input_x, self.l1_weights) + self.l1_biases
        self.l1_predictions = sigmoid(self.l1_biased_vals)
        
        self.l2_biased_vals = np.dot(self.l1_predictions, self.l2_weights) + self.l2_biases
        self.final_pred = softmax(self.l2_biased_vals)
        return self.final_pred

    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        # given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        derivative_wrt_preds = self.final_pred - np.identity(self.final_pred.shape[1])[label]

        l2_derivative_wrt_biased_vals = self.final_pred*(1-self.final_pred) * derivative_wrt_preds
        l2_derivative_wrt_weights = np.dot(l2_derivative_wrt_biased_vals.T, self.l1_predictions)
        l2_derivative_wrt_biases = np.sum(l2_derivative_wrt_biased_vals, axis=0)

        l1_derivative_wrt_biased_vals = self.l1_predictions*(1-self.l1_predictions) * np.dot(l2_derivative_wrt_biased_vals, self.l2_weights.T)
        l1_derivative_wrt_weights = np.dot(l1_derivative_wrt_biased_vals.T, input_x)
        l1_derivative_wrt_biases = np.sum(l1_derivative_wrt_biased_vals, axis=0)

        self.l2_weights -= learning_rate * l2_derivative_wrt_weights.T
        self.l2_biases -= learning_rate * l2_derivative_wrt_biases
        self.l1_weights -= learning_rate * l1_derivative_wrt_weights.T
        self.l1_biases -= learning_rate * l1_derivative_wrt_biases
