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
import sys

np.random.seed(1234)

def make_train(input_size,output_size,mem_size,mem_width,hidden_size=100):
	P = Parameters()

        # Build controller. ctrl is a network that takes an external and read input
        # and returns the output of the network and its hidden layer
	ctrl = controller.build(P,input_size,output_size,mem_size,mem_width,hidden_size)

        # Build model that predicts output sequence given input sequence
	predict = model.build(P,mem_size,mem_width,hidden_size,ctrl)

	input_seq = T.matrix('input_sequence')
	output_seq = T.matrix('output_sequence')
        [M,weights,output_seq_pred] = predict(input_seq)

        # Setup for adadelta updates
	cross_entropy = T.sum(T.nnet.binary_crossentropy(5e-6 + (1 - 2*5e-6)*output_seq_pred,output_seq),axis=1)
	params = P.values()
	l2 = T.sum(0)
	for p in params:
		l2 = l2 + (p ** 2).sum()
	cost = T.sum(cross_entropy) + 1e-3*l2
        # clip gradients
	grads  = [ T.clip(g,-100,100) for g in T.grad(cost,wrt=params) ]

	train = theano.function(
			inputs=[input_seq,output_seq],
			outputs=cost,
			updates=updates.adadelta(params,grads)
		)

	return P,train

if __name__ == "__main__":
	model_out = sys.argv[1]

	P,train = make_train(
		input_size = 8,
		mem_size   = 128,
		mem_width  = 20,
		output_size = 8
	)

	max_sequences = 50000
	patience = 20000
	patience_increase = 3
	improvement_threshold = 0.995
	best_cost = np.inf
	test_cost = 0.
	cost = None
	alpha = 0.95
	for counter in xrange(max_sequences):
                # Start training with short sequences, gradually increase max length
                # as training progresses
		length = np.random.randint(int(20 * (min(counter,25000)/float(25000))**2) +1) + 1
		i,o = tasks.copy(8,length)
		if cost == None: cost = train(i,o)
		else: cost = alpha * cost + (1 - alpha) * train(i,o)
		print "round: ", counter, " cost: ", cost
		if cost < best_cost:
			# improve patience if loss improvement is good enough
			if cost < best_cost * improvement_threshold:
				patience = max(patience, counter * patience_increase)
			P.save(model_out)
			best_cost = cost

		if patience <= counter: break
