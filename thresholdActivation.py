class Neuron:
    def __init__(self, weights, bias, threshold):
        """Initialize the neuron with given weights, bias, and threshold."""
        self.weights = weights
        self.bias = bias
        self.threshold = threshold

    def activate(self, input_sum):
        """Threshold activation function: output 1 if input_sum ≥ threshold, else 0."""
        return 1 if input_sum >= self.threshold else 0

    def predict(self, inputs):
        """Compute weighted sum and pass it through the activation function."""
        input_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.activate(input_sum)


weights = [0.5, 0.7] 
bias = -0.2          
threshold = 0.5       

neuron = Neuron(weights, bias, threshold)


inputs = [1, 1] 
output = neuron.predict(inputs)

print(f"Neuron output: {output}")
