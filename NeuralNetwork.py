import random
import numpy as np
import math

class NeuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.in_hidden_weights = np.random.rand(self.hidden_nodes, self.input_nodes) * 2 - 1

        self.hidden_output_weights = np.random.rand(self.output_nodes, self.hidden_nodes) * 2 - 1

        self.in_hidden_biases = np.random.rand(self.hidden_nodes, 1) * 2 - 1

        self.hidden_out_biases = np.random.rand(self.output_nodes, 1) * 2 - 1


        self.sigmoid_v = np.vectorize(self.sigmoid)

    def sigmoid(self,x):
        return (1/(1+math.exp(-x)))

    def feedforward(self, inputs):

        self.inputs = inputs

        self.hidden_layer = np.dot(self.in_hidden_weights, self.inputs)

        self.hidden_layer = self.sigmoid_v(self.hidden_layer + self.in_hidden_biases)


        self.output_layer = np.dot(self.hidden_output_weights, self.hidden_layer)

        self.output_layer = self.sigmoid_v(self.output_layer + self.hidden_out_biases)


        return self.output_layer

