class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.weights = [0.0] * (input_size + 1) 
        self.lr = lr  
        self.epochs = epochs  

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
       
        summation = self.weights[0]  
        for i in range(len(inputs)):
            summation += inputs[i] * self.weights[i + 1]
        return self.activation_fn(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
              
                self.weights[0] += self.lr * error
            
                for i in range(len(inputs)):
                    self.weights[i + 1] += self.lr * error * inputs[i]


training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]


perceptron = Perceptron(input_size=2)


perceptron.train(training_inputs, labels)


print("Testing AND gate:")
for inputs in training_inputs:
    output = perceptron.predict(inputs)
    print(f"Input: {inputs} -> Output: {output}")