import numpy as np

# activation function
def relu(inputs):
    return np.maximum(inputs, 0)

# output probability distribution function
def softmax(inputs):
    exp = np.exp(inputs)
    return exp/np.sum(exp, axis = 1, keepdims = True)

# loss
def cross_entropy(inputs, y):
    indices = np.argmax(y, axis = 1).astype(int)
    probability = inputs[np.arange(len(inputs)), indices] 
    log = np.log(probability)
    loss = -1.0 * np.sum(log) / len(log)
    return loss

# L2 regularization
def L2_regularization(la, weight1, weight2):
    weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss

#forward pass
def forward_pass(input_image, weight1, bias1, weight2, bias2):
    input_image_flattened = input_image.reshape((1, -1))

    input_layer = np.dot(input_image_flattened, weight1)
    hidden_layer = relu(input_layer + bias1)
    
    scores = np.dot(hidden_layer, weight2) + bias2

    probabilities = softmax(scores)
    return probabilities
