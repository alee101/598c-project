import numpy as np
import random

np.random.seed(1234)
random.seed(1234)


def copy(input_size, sequence_length):
	"""
	Randomly generate a sequence of 1's and 0's for the copy task.
	Returns the input sequence and the expected output sequnece
		input_size: dimension of the data vector to copy
		sequence_length: number of data vectors to copy
	"""

	# Generate a sequence of 1's and 0's with binomial distribution
	sequence = np.random.binomial(1,0.5,(sequence_length,input_size-1)).astype(np.uint8)

	# Initialize the input and output sequences to be zeros
	input_sequence  = np.zeros((sequence_length*2+1,input_size),dtype=np.float32)
	output_sequence = np.zeros((sequence_length*2+1,input_size),dtype=np.float32)

	# Copy the randomly generated sequence to the start of the input sequence
	# and the end of the output sequence. Add the delimiter after the sequence.
	input_sequence[:sequence_length,:-1]  = sequence
	input_sequence[sequence_length,-1] = 1
	output_sequence[sequence_length+1:,:-1] = sequence
	return input_sequence, output_sequence
