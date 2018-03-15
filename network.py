import numpy
import  scipy.special

class neuralNetwork:
    def __init__(self, inputnodes, hidennodes, outputnodes, learningrate):
        self.inputNodes = inputnodes
        self.hidenNodes = hidennodes
        self.outputNodes = outputnodes

        self.learningRate = learningrate

        #self.wih = numpy.random.rand(self.hidenNodes,self.inputNodes) - 0.5
        #self.who = numpy.random.rand(self.outputNodes,self.hidenNodes) - 0.5
        self.wih = numpy.random.normal(0.0, pow(self.hidenNodes, -0.5), (self.hidenNodes, self.inputNodes))
        self.who = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hidenNodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self):

        pass

    def query(self, inputs_list):
        #преобразовать список входящих значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        #расчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        #расчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)


        #расчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #расчитать исходящие сигналы для выходящего слоя
        final_outputs = self.activation_function(final_inputs)

        return  final_outputs
        pass
    pass


