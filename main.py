from network import neuralNetwork
import  numpy
import matplotlib.pyplot as plt
from IPython import get_ipython
ipython = get_ipython()




"""input_nodes = 3
hiden_nodes = 3
output_nodes = 3
learning_rate = 0.3
n = neuralNetwork(input_nodes, hiden_nodes, output_nodes, learning_rate)
print(numpy.random.rand(3,3) - 0.5)
print(n.query([1.0, 0.5, -1.5]))"""




data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines();
data_file.close()

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show(block=True)