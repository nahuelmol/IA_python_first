import numpy as np

def sigmoid(x):
	result = 1/(1+ np.exp(-x))
	return result

def sigmoid_derivative(x):
	result = x * (1 - x)
	return result

training_inputs = np.array([
	[0,0,1],
	[1,1,1],
	[1,0,1],
	[0,1,1]
	])

#the real outputs
training_outputs = np.array([[0,1,1,0]]).T 
np.random.seed(1)

#generating random numbers
synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Random starting synaptic weights:")
print(synaptic_weights)

#loop for adjustment of weights with errors using the real outputs 
for ite in range(100000):
	input_layer = training_inputs
	outputs = sigmoid(np.dot(input_layer,synaptic_weights))
	#at the first iteration
	#the outputs are very random cause they don't have any relations with the real outputs

	error = training_outputs - outputs
	adjustments = error + sigmoid_derivative(outputs)

	synaptic_weights += np.dot(input_layer.T,adjustments)

print("New weights: ")
print(synaptic_weights)

print("Outputs are training:")
print(outputs)