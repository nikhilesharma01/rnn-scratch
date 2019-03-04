import numpy as np 
import itertools
from datetime import datetime
import sys

vocabulary_size = 8000

# Get training data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Initialize parameters
class RNN():
	def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 4):
		# Initialize instance variables
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate

		# Initialize randomly the matrices U, V and W
		self.U = self.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.V = self.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
		self.W = self.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))

	# Forward propagation
	def softmax(x):
		xt = np.exp(x - np.max(x))
		return xt / np.sum(xt)

	def forward_propagation(self, X):
		# No. of timesteps
		T = len(X)

		# Save all hidden stages of s during forward propagation
		# s_t = U.dot(x_t) + W.dot(s_t-1)

		s = np.zeros((T+1, self.hidden_dim))

		# Initialize s[-1] and the output to all zeros
		s[-1] = np.zeros(self.hidden_dim)
		o = np.zeros((T, self.word_dim))

		for t in np.arange(T):
			s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
			o[t] = softmax(self.V.dot(s[t]))

		return [o, s]
	RNN.forward_propagation = forward_propagation

	# Predict the next word
	def predict(self, X):
		o, s = self.forward_propagation(X)
		return np.argmax(o, axis = 1)

	RNN.predict = predict


	# Calculate the loss
	# Using cross-entropy loss for loss calculation
	def calculate_total_loss(self, X, y):
		loss = 0

		for i in np.arange(len(y)):
			o, s = self.forward_propagation(X)
			correct_predictions = o[np.arange(len(y[i])), y[i]]

			# Add the cross-entropy loss
			loss += -1 * np.sum(np.log(correct_predictions))

		return loss


	def calculate_loss(self, X, y):
		N = np.sum((len(y_i) for y_i in y))
		return self.calculate_total_loss / N

	RNN.calculate_total_loss = calculate_total_loss
	RNN.calculate_loss = calculate_loss


	# Perform Backpropagation through time to 
	# propagate the error backwards

	def bptt(self, X, y):
		T = len(y)

		# Perform forward propagation
		o, s =  self.forward_propagation(X)

		# Initialize the gradients to all zeros
		dLdU = np.zeros(self.U.shape)
		dLdV = np.zeros(self.V.shape)
		dLdW = np.zeros(self.W.shape)

		delta_o = o
		# y_hat - y
		delta_o[np.arange(len(y)), y] -= 1

		# Move backwards for each output
		for t in np.arange(T):
			dLdV += np.outer(delta_o, s[t].T)

			# Initial delta calculation 
			# d3, d2, etc. in the notes

			delta_t = self.V.dot(delta_o[t]) * (1 - s[t] ^ 2)

			# Given time step t, go back to t-1, t-2, ...
			for bptt_step in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
				dLdW += np.outer(delta_o[t], s[bptt_step - 1])
				dLdU[:, X[bptt_step]] += delta_t

				delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step -1] ^ 2)

		return [dLdU, dLdV, dLdW]

	RNN.bptt = bptt

	# Implement gradient check
	# Usually an expensive operation
	# Check only on small vocabulary

	def gradient_check(self, X, y, h = 0.001, error_threshold = 0.01):
		# Calculate gradient using backpropagation
		bptt_gradients = self.bptt(X, y)

		# List all parameters we went to check
		model_parameters = ["U", "V", "W"]

		# Check gradient for each parameter
		for pidx, pname in enumerate(model_parameters):
			# Get the actual parameter value from model_parameter
			parameter = operator.attrgetter(pname)(self)
			print("performing gradient check for parameter %s with size %d. " %(pname, np.prod(parameter.shape)))

			# Iterate over each element of the parameter matrix
			it = np.nditer(parameter, flags = ['multi_index'], op_flags = ['readwrite'])

			while not it.finished:
				ix = it.multi_index
				# Save original value
				original_value = parameter[ix]

				# Estimate gradient
				parameter[ix] = original_value + h
				grad_plus = self.calculate_total_loss([X], [y])
				parameter[ix] = original_value - h
				grad_minus = self.calculate_total_loss([X], [y])

				estimated_gradient = (grad_plus - grad_minus) / (2 * h)
				parameter[ix] = original_value

				backprop_gradient = bptt_gradients[pidx][ix]

				# Calculate relative error
				relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))

				# If error is too large, fail the gradient check
				if relative_error < error_threshold:
					print("Gradient check error: parameter = %s ix = %s" %(pname, ix))
                	print("+h Loss: %f" % gradplus)
                	print("-h Loss: %f" % gradminus)
                	print("Estimated gradient: %f" % estimated_gradient)
                	print("Backpropagation gradient: %f" % backprop_gradient)
                	print("Relative error: %f" % relative_error)
                	return
                it.iternext()
            print("Gradient check for parameter %s passed. " %(pname))

    RNN.gradient_check = gradient_check

    def numpy_sgd_step(self, x, y, learning_rate):
    	dLdU, dLdV, dLdW = self.bptt(x, y)
    	self.U -= learning_rate * dLdU
    	self.V -= learning_rate * dLdV
    	self.W -= learning_rate * dLdW

    RNN.sgd_step = numpy_sgd_step


# Train the model and make predictions on training data
np.random.seed(10)
model = RNN(vocabulary_size)
o, s = model.forward_propagation(X_train[100])
print (o.shape)
print (o)

predictions = model.predict(X_train[100])
print (predictions.shape)
print (predictions)

print("Expected Loss for random prediction: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))

# Perform a gradient check
grad_check_vocab_size = 100
np.random.seed(10)
model = RNN(grad_check_vocab_size, 10, bptt_truncate = 1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

def train_with_sgd(model, X_train, y_train, learning_rate = 0.005, nepoch = 100, evaluate_loss_after = 5):
    # keep track of the losses so that we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: loss after num_examples_seen=%d epoch=%d: %f" %(time, num_examples_seen, epoch, loss))
            # adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("setting learning rate to %f" %(learning_rate))
            sys.stdout.flush()
        # for each training example...
        for i in range(len(y_train)):
            # one sgd step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


np.random.seed(10)
model = RNNNumpy(vocabulary_size)
%timeit model.sgd_step(X_train[10], y_train[10], 0.005)

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch = 10, evaluate_loss_after = 1)