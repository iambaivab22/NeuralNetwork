# # Training Data
# X = [[5.9, 75], [5.8, 86], [5.2, 50], [5.4, 55], [6.1, 85], [5.5, 62]]
# y = [1, 1, 0, 0, 1, 0]  # Male: 1, Female: 0

# # Initialize weights and bias
# weights = [0, 0]
# bias = 0
# learning_rate = 0.1
# epochs = 1000

# # Train Perceptron Model
# for _ in range(epochs):
#     for i in range(len(X)):
#         linear_output = sum(X[i][j] * weights[j] for j in range(2)) + bias
#         prediction = 1 if linear_output >= 0 else 0
#         error = y[i] - prediction
#         for j in range(2):
#             weights[j] += learning_rate * error * X[i][j]
#         bias += learning_rate * error

# # Prediction Function
# def predict(sample):
#     linear_output = sum(sample[j] * weights[j] for j in range(2)) + bias
#     return 1 if linear_output >= 0 else 0

# # Test Data
# test_samples = [[6, 82], [5.3, 52]]
# for sample in test_samples:
#     prediction = predict(sample)
#     label = "Male" if prediction == 1 else "Female"
#     print(f"Input {sample} is classified as {label}")



# Training Data (Height, Weight) and Labels (Male=1, Female=0)
X = [
    [5.9, 75],
    [5.8, 86],
    [5.2, 50],
    [5.4, 55],
    [6.1, 85],
    [5.5, 62]
]

y = [1, 1, 0, 0, 1, 0]  # Male = 1, Female = 0

# Min-Max Scaling (Manual)
def min_max_scaling(X):
    min_vals = [min(col) for col in zip(*X)]  # Column-wise min
    max_vals = [max(col) for col in zip(*X)]  # Column-wise max
    
    scaled_X = [
        [(X[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]) for j in range(len(X[0]))]
        for i in range(len(X))
    ]
    
    return scaled_X, min_vals, max_vals

# Scale Training Data
X_scaled, X_min, X_max = min_max_scaling(X)

# Perceptron Class (Pure Python)
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [0.5] * (len(X_scaled[0]) + 1)  # Initialize weights (including bias)

    def activation(self, x):
        return 1 if x >= 0 else 0  # Step function

    def dot_product(self, weights, inputs):
        return sum(w * i for w, i in zip(weights, inputs))

    def train(self, X, y):
        # Add bias term (1) to inputs
        X = [[1] + row for row in X]
        
        for _ in range(self.epochs):
            for i in range(len(X)):
                z = self.dot_product(self.weights, X[i])
                y_pred = self.activation(z)
                error = y[i] - y_pred
                # Update weights
                self.weights = [w + self.learning_rate * error * x_i for w, x_i in zip(self.weights, X[i])]

    def predict(self, X):
        X = [[1] + row for row in X]  # Add bias term
        return [self.activation(self.dot_product(self.weights, x)) for x in X]

# Train Perceptron
perceptron = Perceptron()
perceptron.train(X_scaled, y)

# Scale New Inputs
def scale_new_data(X_new, X_min, X_max):
    return [
        [(X_new[i][j] - X_min[j]) / (X_max[j] - X_min[j]) for j in range(len(X_new[0]))]
        for i in range(len(X_new))
    ]

# New Data to Predict
test_data = [[6, 82], [5.3, 52]]
test_data_scaled = scale_new_data(test_data, X_min, X_max)

# Predictions
predictions = perceptron.predict(test_data_scaled)

# Output Results
for i, test in enumerate(test_data):
    print(f"Input: {test}, Predicted Class: {'Male' if predictions[i] == 1 else 'Female'}")
