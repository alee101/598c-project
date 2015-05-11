import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U

def build(P,input_size,mem_width,mem_size,shift_width):
	P["W_%d_shift"] = U.initial_weights(input_size,shift_width)
	P["b_%d_shift"] = 0. * U.initial_weights(shift_width)

	P["W_%d_gamma"] = U.initial_weights(input_size)
	P["b_%d_gamma"] = 0.
	P["W_%d_g"]     = U.initial_weights(input_size)
	P["b_%d_g"]     = 0.

	P["W_%d_erase"] = U.initial_weights(input_size,mem_width)
	P["b_%d_erase"] = 0. * U.initial_weights(mem_width)
	P["W_%d_add"]   = U.initial_weights(input_size,mem_width)
	P["b_%d_add"]   = 0. * U.initial_weights(mem_width)


	def head_params(x):
		# shift
		shift_t = U.vector_softmax(T.dot(x,P["W_%d_shift"]) + P["b_%d_shift"])
		shift_t.name = "shift_t"

		# scalars
		_gamma_t = T.dot(x,P["W_%d_gamma"]) + P["b_%d_gamma"]
		gamma_t = T.nnet.softplus(_gamma_t) + 1.
#		gamma_t = (_gamma_t > 0)*_gamma_t + 1.
#		gamma_t = T.exp(_gamma_t) + 1.

		g_t     = T.nnet.sigmoid(T.dot(x,P["W_%d_g"]) + P["b_%d_g"])

		erase_t = T.nnet.sigmoid(T.dot(x,P["W_%d_erase"]) + P["b_%d_erase"])
		add_t   = T.dot(x,P["W_%d_add"]) + P["b_%d_add"]

		return g_t,shift_t,gamma_t,erase_t,add_t
	return head_params
