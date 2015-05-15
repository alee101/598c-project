import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U

def build(P,input_size,mem_width,mem_size,shift_width):
        """
        NTM heads are implemented as another hidden layer coming after
        the last hidden layer of the controller that emits
        k_t, beta_t, g_t, s_t, gamma_t as outputs (see Controller outputs
        of Figure 2 in paper) along with erase and add vectors
        """
	P["W_key"]   = U.initial_weights(input_size,mem_width)
	P["b_key"]   = 0. * U.initial_weights(mem_width)

        P["W_beta"]  = 0. * U.initial_weights(input_size)
        P["b_beta"]  = 0.

	P["W_shift"] = U.initial_weights(input_size,shift_width)
	P["b_shift"] = 0. * U.initial_weights(shift_width)

	P["W_g"]     = U.initial_weights(input_size)
	P["b_g"]     = 0.

	P["W_erase"] = U.initial_weights(input_size,mem_width)
	P["b_erase"] = 0. * U.initial_weights(mem_width)
	P["W_add"]   = U.initial_weights(input_size,mem_width)
	P["b_add"]   = 0. * U.initial_weights(mem_width)


	def head_params(x):
                """
                Takes hidden layer from controller computes
                k_t, beta_t, g_t, s_t, gamma_t, and erase and add
                vectors as outputs
                """
		# key
		key_t = T.dot(x,P["W_key"]) + P["b_key"]

                # key strength
		_beta_t  = T.dot(x,P["W_beta"])  + P["b_beta"]
		beta_t  = T.nnet.softplus(_beta_t)

                # interpolation gate
		g_t     = T.nnet.sigmoid(T.dot(x,P["W_g"]) + P["b_g"])

		# shift
		shift_t = U.vector_softmax(T.dot(x,P["W_shift"]) + P["b_shift"])
		shift_t.name = "shift_t"

                # erase and add vectors
		erase_t = T.nnet.sigmoid(T.dot(x,P["W_erase"]) + P["b_erase"])
		add_t   = T.dot(x,P["W_add"]) + P["b_add"]

		return key_t,beta_t,g_t,shift_t,,erase_t,add_t
	return head_params
