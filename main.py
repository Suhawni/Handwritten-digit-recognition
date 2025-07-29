import numpy as np
from network import Network
import mnist

def shuffle_mnist(images, labels):
    combined = list(zip(images, labels))
    np.random.shuffle(combined)
    shuffled_images, shuffled_labels = zip(*combined)
    shuffled_images = np.array(shuffled_images)
    shuffled_labels = np.array(shuffled_labels)
    return shuffled_images, shuffled_labels

# load data
num_classes = 10
train_images = mnist.train_images() 
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print("Shuffling training data...")
train_images, train_labels = shuffle_mnist(train_images, train_labels)

print("Shuffling test data...")
test_images, test_labels = shuffle_mnist(test_images, test_labels)


X_train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_train = X_train / 255 
y_train = np.eye(num_classes)[train_labels] 
X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_test = X_test / 255 
y_test = test_labels

net = Network(
                 num_nodes_in_layers = [784, 20, 10], 
                 batch_size = 1,
                 num_epochs = 5,
                 learning_rate = 0.001, 
                 weights_file = 'weights.pkl',
             )

print("Training...")
net.train(x_train, y_train)

print("Testing...")
net.test(x_test, y_test)
