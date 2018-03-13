import numpy

class neuralNetwork:
    def __init__(self,inputnodes, hidennodes, outputnodes, learningrate):
        self.inputNodes = inputnodes
        self.hidenNodes = hidennodes
        self.outputNodes = outputnodes
        self.learningRate = learningrate

        #self.wih = numpy.random.rand(self.hidenNodes,self.inputNodes) - 0.5
        #self.who = numpy.random.rand(self.outputNodes,self.hidenNodes) - 0.5

        self.wih = numpy.random.normal(0.0, pow(self.hidenNodes, -0.5), (self.hidenNodes, self.inputNodes))
        self.who = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hidenNodes))
        pass

    def train(self):

        pass

    def query(self):

        pass
    pass


