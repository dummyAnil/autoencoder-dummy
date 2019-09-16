import model
from PIL import Image
from mlxtend.data import loadlocal_mnist
import numpy as np
import sys

TEST_DATA_IMAGES_PATH = "/home/ubuntu/school/409-Makeup/Autoencoder/samples/t10k-images.idx3-ubyte"
TEST_DATA_LABELS_PATH = "/home/ubuntu/school/409-Makeup/Autoencoder/samples/t10k-labels.idx1-ubyte"

X_test, Y_test = loadlocal_mnist(
    images_path=TEST_DATA_IMAGES_PATH,
    labels_path=TEST_DATA_LABELS_PATH)

X_test = X_test / 255

FILE_PATH = sys.argv[1]
MODEL_NANE = FILE_PATH.split("/")[-1]

nn = model.NN()
params = nn.load(FILE_PATH)
deep_architecture = model.architectures[model.DEEP]
vanilla_architecture = model.architectures[model.VANILLA]


if MODEL_NANE == model.VANILLA + ".txt":
    for i in range(20, 25):
        test_X = X_test[i].reshape(784, 1)
        prediction, cashe = nn.full_forward_propagation(test_X, params, vanilla_architecture)

        img = Image.fromarray((test_X.reshape(28, 28) * 256).astype(np.uint8))
        img.save("test" + model.VANILLA + str(i) + ".png")

        img = Image.fromarray((prediction.reshape(28, 28) * 256).astype(np.uint8))
        img.save("prediction" + model.VANILLA + str(i) + ".png")


elif MODEL_NANE == model.DEEP + ".txt":
    for i in range(20, 25):
        test_X = X_test[i].reshape(784, 1)
        prediction, cashe = nn.full_forward_propagation(test_X, params, deep_architecture)

        img = Image.fromarray((test_X.reshape(28, 28) * 256).astype(np.uint8))
        img.save("test" + model.DEEP + str(i) + ".png")

        img = Image.fromarray((prediction.reshape(28, 28) * 256).astype(np.uint8))
        img.save("prediction" + model.DEEP + str(i) + ".png")



