import sys
import time
import tkinter

import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

import model

TRAIN_DATA_IMAGES = "/train-images.idx3-ubyte"
TRAIN_DATA_LABELS = "/train-labels.idx1-ubyte"

DATA_PATH = sys.argv[1]

X_train, Y_train = loadlocal_mnist(
    images_path=DATA_PATH + TRAIN_DATA_IMAGES,
    labels_path=DATA_PATH + TRAIN_DATA_LABELS)

print('Dimensions: %s x %s' % (X_train.shape[0], X_train.shape[1]))
print("Dimensions of one image %s" % X_train[0].shape)

# Normalize values
X_train = X_train / 255

# Loss functions
EUCLIDEAN_DISTANCE = model.EUCLIDEAN_DISTANCE
CROSS_ENTROPY = model.CROSS_ENTROPY

# Architecture parameter
EPOCH = 50
LEARNING_RATE = 0.01
NUMBER_OF_TRAINING_SAMPLES = 5000

neural_network = model.NN()
deep_architecture = model.architectures[model.DEEP]
vanilla_architecture = model.architectures[model.VANILLA]

vanilla_start = time.time()
print("-----Start Vanilla Model Training-----")
vanilla_params, vanilla_loss_history, vanilla_loss = neural_network.train(X_train, vanilla_architecture, EPOCH,
                                                                          LEARNING_RATE,
                                                                          NUMBER_OF_TRAINING_SAMPLES, EUCLIDEAN_DISTANCE)
vanill_end = time.time()
neural_network.save(vanilla_params, model.VANILLA)

plt.plot(range(EPOCH), vanilla_loss_history, color='red')
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.title("Loss History")
plt.show()


deep_start = time.time()
print("-----Start Deep Model Training-----")
deep_params, deep_loss_history, deep_loss = neural_network.train(X_train, deep_architecture, EPOCH, LEARNING_RATE,
                                                                 NUMBER_OF_TRAINING_SAMPLES, CROSS_ENTROPY)
deep_end = time.time()
neural_network.save(deep_params, model.DEEP)

plt.plot(range(EPOCH), deep_loss_history, color='red')
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.title("Loss History")
plt.show()

print("The Architecture:")
for i in deep_architecture:
    print(i)
print("Number of training samples: ", NUMBER_OF_TRAINING_SAMPLES)
print("Learning Rate: ", LEARNING_RATE)
print("Epochs: ", EPOCH)
print("-----Results for Vanilla Model-----")
print("Loss: ", vanilla_loss)
print("Loss Function: ", CROSS_ENTROPY)
print("Elapsed time during training: %s minutes" % ((vanill_end - vanilla_start) / 60))
print(print("-----Results for Deep Model-----"))
print("Loss: ", deep_loss)
print("Loss Function: ", CROSS_ENTROPY)
print("Elapsed time during training: %s minutes" % ((deep_end - deep_start) / 60))
