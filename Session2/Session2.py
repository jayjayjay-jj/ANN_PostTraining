import numpy as np
from random import choice

dataset = [
    # XOR 
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0]))
]

num_of_input = 2
num_of_output = 1

# Firstly give the weight random value
weight = np.random.normal(size=[num_of_input, num_of_output])

# Bias = the addition/ new input (Somekind of distraction) -> Only to output
bias = np.random.normal(size=[num_of_output])

def activation(output):
    if output >= 0:
        return 1
    return 0

# Combining
def feed_forward(feature):
    # Formula = x1 * w1 + x2 * w2 + ... + xn * wn 

    # np.matmul -> matrix multiplication
    output = np.matmul(feature, weight) + bias
    return activation(output)

# Looping the activation and feed_forward to train the data using input
# Epoch = how many time will we train the model using the same data
epoch = 1000

# How fast the model study, the bigger the learning_rate, the worse the weight will be, the smaller the learning_rate is, the better the weight will be (Commonly)
learning_rate = 0.1

for i in range(1, epoch+1):
    # We use dataset as both the feature and target (The dataset is tuple, so we use both feature and target)
    feature, target = choice(dataset)  

    output = feed_forward(feature)
    error = target - output

    # Update weight and bias after every training (This is based on the formula)
    # The last updated weight is the knowledge
    weight = weight + learning_rate * error * feature.reshape(num_of_input, num_of_output)
    bias = bias + learning_rate * error

    # Print the accuracy for every range of epoch
    # For example, this is printing the accurange for every 20 epoch
    if (i % 20 == 0):
        correct = 0
        
        # Calculate the accuracy
        for feature, target in dataset:
            output = feed_forward(feature)

            if target == output:
                correct += 1

        print(f'Epoch {i}, Accuracy = {correct / len(dataset) * 100}%')

