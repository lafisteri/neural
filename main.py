import  numpy
from network import neuralNetwork

input_nodes = 3
hiden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = neuralNetwork(input_nodes, hiden_nodes, output_nodes, learning_rate)

#print(numpy.random.rand(3,3) - 0.5)


print(n.query([1.0, 0.5, -1.5]))