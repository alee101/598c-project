import theano
import theano.tensor as T
import numpy         as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import controller
import head
import scipy

def cosine_sim(k,M):
        """
        Cosine similarity used for content addressing
        """
	k_unit = k / ( T.sqrt(T.sum(k**2)) + 1e-5 )
	k_unit = k_unit.dimshuffle(('x',0))
	k_unit.name = "k_unit"
	M_lengths = T.sqrt(T.sum(M**2,axis=1)).dimshuffle((0,'x'))
	M_unit = M / ( M_lengths + 1e-5 )

	M_unit.name = "M_unit"

	return T.sum(k_unit * M_unit,axis=1)

def build_step(P,controller,controller_size,mem_size,mem_width,similarity=cosine_sim,shift_width=3):
        # Set of shift indices (for shift_width=3, have shift offsets of -1, 0, and +1)
	shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[np.arange(-(shift_width//2),(shift_width//2)+1)][::-1]

        # Initial N X M memory: M_0
	P.memory_init = 2 * (np.random.rand(mem_size,mem_width) - 0.5)
	memory_init = P.memory_init

        # Initial N-dim weight vector: w_0
	P.weight_init = np.random.randn(mem_size)
	weight_init = U.vector_softmax(P.weight_init)

        # heads is a function taking the hidden layer of the controller and
        # computes the key, key strength, interpolation gate,
        # sharpening factor, and erase and add vectors as outputs
        heads = head.build(P,controller_size,mem_width,mem_size,shift_width)

	def build_memory_curr(M_prev,erase_head,add_head,weight):
                """
                Update memory with write consisting of erase and add
                (described in section 3.2 in paper)
                """
		weight = weight.dimshuffle((0,'x'))

		erase_head = erase_head.dimshuffle(('x',0))
		add_head   = add_head.dimshuffle(('x',0))

                # Equation (3)
		M_erased = M_prev   * (1 - (weight * erase_head))
                # Equation (4)
		M_curr   = M_erased +      (weight * add_head)

		return M_curr

	def build_read(M_curr,weight_curr):
                """
                Obtain read vector r_t (Equation (2) in paper)
                """
		return T.dot(weight_curr, M_curr)

	def shift_convolve(weight,shift):
                """
                Circular convolution (Equation (8) in paper)
                """
		shift = shift.dimshuffle((0,'x'))
		return T.sum(shift * weight[shift_conv],axis=0)

	def build_head_curr(weight_prev,M_curr,head,input_curr):
		"""
		This function is best described by Figure 2 in the paper.
		"""
                # input_curr is final hidden layer from controller
                # this is passing the hidden layer into head_params of head.py
		g,shift,gamma,erase,add = head(input_curr)

                weight_c = np.zeros((mem_size,))
                weight_c[0] = 1
                #weight_c = T.as_tensor_variable(weight_c)
                weight_c = theano.shared(weight_c)
		weight_c.name = "weight_c"

		# 3.3.2 Focusing by Location (Equation (7))
		weight_g       = g * weight_c + (1 - g) * weight_prev
		weight_g.name = "weight_g"

                # Equation (8)
		weight_shifted = shift_convolve(weight_g,shift)

                # Equation (9)
		weight_sharp   = weight_shifted ** gamma
		weight_curr    = weight_sharp / T.sum(weight_sharp)

		return weight_curr,erase,add

	def step(input_curr,M_prev,weight_prev):
                """
                Update the weights and memory from the previous time step
                given the current input
                """
                # Get read vector r_t
		read_prev = build_read(M_prev,weight_prev)

                # Feed current input and read input to controller to get
                # controller output and hidden layer of controller
		output,controller_hidden = controller(input_curr,read_prev)

                # Obtain new weight vector (as described in figure 2) and erase and add vectors
                weight_curr,erase,add = build_head_curr(weight_prev,M_prev,heads,controller_hidden)
                # Update memory with current weight, erase, and add vectors (Section 3.2 in paper)
                M_curr = build_memory_curr(M_prev,erase,add,weight_curr)

		return M_curr,weight_curr,output
	return step,[memory_init,weight_init,None]

def build(P,mem_size,mem_width,controller_size,ctrl):
        """
        Build model for prediction.
        """
        # step is a function that takes the current external input and state (memory and weight vector)
        # and returns an updated memory and state
        # outputs_info consists of the initial memory and weights
	step,outputs_info = build_step(P,ctrl,controller_size,mem_size,mem_width)

	def predict(input_sequence):
                """
                Use NTM to predict outputs given input_sequence.
                """
		outputs,_ = theano.scan(
				step, # apply step to input_sequence
				sequences    = [input_sequence],
				outputs_info = outputs_info
			)
                # output is current memory, weight, and output (from step)
		return outputs
	return predict
