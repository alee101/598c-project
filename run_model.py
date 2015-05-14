import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U
from theano_toolkit import hinton
import controller
import model
import tasks
import random
import math

def make_model(
		input_size=8,
		output_size=8,
		mem_size=128,
		mem_width=20,
		hidden_size=100):
	"""
	Given the model parameters, return a Theano function for the NTM's model
	"""

	P = Parameters()

	# Build the controller and the read/write head
        # FILL IN HERE

	# Return a Theano function for the NTM
        # FILL IN HERE

	return P,test_fun
