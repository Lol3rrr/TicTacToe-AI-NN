from numpy import exp, array, random, dot

class NeuronLayer():
  def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
    #Seed the random number generator
    #random.seed(1)
    
    self.neuronCount = number_of_neurons
    self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
  def __init__(self, inputCount, inputNeurons, outputCount, hiddenLayers):
    self.layers = []
    
    # adding the first layer as the input layer
    self.layers.append(NeuronLayer(inputNeurons, inputCount))

    for iteration in range(len(hiddenLayers)):
      # adding all hidden layers
      self.layers.append(NeuronLayer(hiddenLayers[iteration], self.layers[len(self.layers) - 1].neuronCount))

    # adding the output layer
    self.layers.append(NeuronLayer(outputCount, self.layers[len(self.layers) - 1].neuronCount))

  # The Sigmoid function, which describes an S shaped curve.
  # We pass the weighted sum of the inputs through this function to
  # normalise them between 0 and 1.
  def __sigmoid(self, x):
    return 1 / (1 + exp(-x))

  # The derivative of the Sigmoid function.
  # This is the gradient of the Sigmoid curve.
  # It indicates how confident we are about the existing weight.
  def __sigmoid_derivative(self, x):
    return x * (1 - x)

  # We train the neural network through a process of trial and error.
  # Adjusting the synaptic weights each time.
  def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
      # Pass the training set through our neural network
      thinkingOutputs = self.think(training_set_inputs)

      errors = []
      deltas = []

      # iterate over the output list in reverse
      for outputIterator in range(len(thinkingOutputs)):
        outputIndex = len(thinkingOutputs) - 1 - outputIterator

        # the error of the 2. layer
        if outputIndex == len(thinkingOutputs) - 1:
          errors.append(training_set_outputs - thinkingOutputs[outputIndex])
        else:
          errors.append(deltas[outputIterator - 1].dot(self.layers[len(self.layers) - outputIterator].synaptic_weights.T))

        # the delta of the 2. layer
        deltas.append(errors[outputIterator] * self.__sigmoid_derivative(thinkingOutputs[outputIndex]))

      # Calculate how much to adjust the weights by
      adjustments = []

      for adjustmentIterator in range(len(deltas)):
        deltaIndex = len(deltas) - 1 - adjustmentIterator
        outputIndex = adjustmentIterator - 1

        if deltaIndex == len(deltas) - 1:
          adjustments.append(training_set_inputs.T.dot(deltas[deltaIndex]))
        else:
          adjustments.append(thinkingOutputs[outputIndex].T.dot(deltas[deltaIndex]))

      # Adjust the weights
      for iterator in range(len(adjustments)):
        self.layers[iterator].synaptic_weights += adjustments[iterator]

  # The neural network thinks.
  def think(self, inputs):
    outputs = []

    for iteration in range(len(self.layers)):
      layerInput = inputs
      if len(outputs) > 0:
        layerInput = outputs[len(outputs) - 1]
      
      layerOutput = self.__sigmoid(dot(layerInput, self.layers[iteration].synaptic_weights))
      outputs.append(layerOutput)

    return outputs

  def getOutput(self, inputs):
    outputs = []

    for iteration in range(len(self.layers)):
      layerInput = inputs
      if len(outputs) > 0:
        layerInput = outputs[len(outputs) - 1]
      
      layerOutput = self.__sigmoid(dot(layerInput, self.layers[iteration].synaptic_weights))
      outputs.append(layerOutput)

    return outputs[len(outputs) - 1]