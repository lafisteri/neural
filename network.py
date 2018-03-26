import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputNodes = inputnodes
        self.hiddenNodes = hiddennodes
        self.outputNodes = outputnodes

        self.learningRate = learningrate

        self.wih = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.who = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, input_list, targets_list):
        # преобразовать список входящих значений в двумерный массив
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # расчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # расчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # расчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # расчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибки выходного слоя = целевое значение - фактическое значение
        output_errors = targets - final_outputs
        # ошибки скрытого слоя это ошибки output_errors расределенные пропорционально
        # весовым коэфициентам связей и рекомбинированы на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты для связей между скрытым и выходным слоями
        self.who += self.learningRate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                  numpy.transpose(hidden_outputs))

        # обновить весовые коэфициенты для связей между входным и скрытым слоями
        self.wih += self.learningRate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                  numpy.transpose(inputs))
        pass

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входящих значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # расчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # расчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # расчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # расчитать исходящие сигналы для выходящего слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass
    pass


