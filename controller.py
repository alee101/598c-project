import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U

import controller
import model
import head
from collections import namedtuple

def build(P,input_size,output_size,mem_size,mem_width,layer_size):
	"""
	Create controller function for use during scan op
	"""
        # Weights for external input
	P.W_input_hidden = U.initial_weights(input_size,layer_size)

        # Weights for input from read head (read from memory)
	P.W_read_hidden  = U.initial_weights(mem_width, layer_size)

        # Shared bias for external input and read head input
	P.b_hidden_0 = 0. * U.initial_weights(layer_size)

        # Weights and biases for output of controller
	P.W_hidden_output = 0. * U.initial_weights(layer_size,output_size)
	P.b_output = 0. * U.initial_weights(output_size)

	def controller(input_t,read_t):
                """
                Controller consists of single hidden layer between inputs and outputs
                """
		hidden_layer = T.tanh(
                        T.dot(input_t,P.W_input_hidden) +\
                        T.dot(read_t,P.W_read_hidden) +\
                        P.b_hidden_0
                )

		output_t = T.nnet.sigmoid(T.dot(hidden_layer,P.W_hidden_output) + P.b_output)

                # Return output and hidden layer of controller used by heads (in model.py)
		return output_t,hidden_layer
	return controller
