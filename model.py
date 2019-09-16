# coding: utf-8
import numpy as np

# Model path
MODEl_PATH = "model.txt"

# Activation functions
SIGMOID = "sigmoid"
RELU = "relu"
TANH = "tanh"

# Loss functions
EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
CROSS_ENTROPY = "CROSS_ENTROPY"

DEEP = "Deep"
VANILLA = "Vanilla"

deep_architecture = [
    {"input_dim": 784, "output_dim": 128, "activation": SIGMOID},
    {"input_dim": 128, "output_dim": 64, "activation": SIGMOID},
    {"input_dim": 64, "output_dim": 784, "activation": SIGMOID}]

vanilla_architecture = [{"input_dim": 784, "output_dim": 128, "activation": SIGMOID},
                        {"input_dim": 128, "output_dim": 784, "activation": SIGMOID}]

architectures = {DEEP: deep_architecture, VANILLA: vanilla_architecture}


class NN:
    def init_layers(self, nn_architecture, seed=99):
        # random seed initiation
        np.random.seed(seed)
        # number of layers in our neural network
        number_of_layers = len(nn_architecture)
        # parameters storage initiation
        params_values = {}

        # iteration over network layers
        for idx, layer in enumerate(nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1

            # extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # initiating the values of the W matrix
            # and vector b for subsequent layers
            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1

        return params_values

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation=RELU):
        # calculation of the input value for the activation function
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        # selection of activation function
        if activation is RELU:
            activation_func = Functions.relu
        elif activation is SIGMOID:
            activation_func = Functions.sigmoid
        elif activation is TANH:
            activation_func = Functions.tanh
        else:
            raise Exception('Non-supported activation function')

        # return of calculated activation A and the intermediate Z matrix
        return activation_func(Z_curr), Z_curr

    def full_forward_propagation(self, X, params_values, nn_architecture):
        # creating a temporary memory to store the information needed for a backward step
        memory = {}
        # X vector is the activation for layer 0 
        A_curr = X

        # iteration over network layers
        for idx, layer in enumerate(nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1
            # transfer the activation from the previous iteration
            A_prev = A_curr

            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]
            # extraction of W for the current layer
            W_curr = params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = params_values["b" + str(layer_idx)]
            # calculation of activation for the current layer
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            # saving calculated values in the memory
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        # return of prediction vector and a dictionary containing intermediate values
        return A_curr, memory

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation=RELU):
        # number of examples
        m = A_prev.shape[1]

        # selection of activation function
        if activation is RELU:
            backward_activation_func = Functions.relu_prime
        elif activation is SIGMOID:
            backward_activation_func = Functions.sigmoid_prime
        elif activation is TANH:
            backward_activation_func = Functions.tanh_prime
        else:
            raise Exception('Non-supported activation function')

        # calculation of the activation function derivative
        dZ_curr = backward_activation_func(dA_curr, Z_curr)

        # derivative of the matrix W
        dW_curr = np.dot(dZ_curr, A_prev.T)
        # derivative of the vector b
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True)
        # derivative of the matrix A_prev
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, Y_hat, Y, memory, params_values, nn_architecture, loss=EUCLIDEAN_DISTANCE,
                                  loss_function=None):
        grads_values = {}
        # number of examples
        m = Y.shape[1]
        # a hack ensuring the same shape of the prediction vector and labels vector
        Y = Y.reshape(Y_hat.shape)

        # initiation of gradient descent algorithm(Calculate loss)
        if loss is EUCLIDEAN_DISTANCE:
            loss_function = Functions.euclidean_distance_prime
        elif loss is CROSS_ENTROPY:
            loss_function = Functions.cross_entropy_prime

        dA_prev = loss_function(Y_hat, Y)

        for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
            # we number network layers from 1
            layer_idx_curr = layer_idx_prev + 1
            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def update(self, params_values, grads_values, nn_architecture, learning_rate):
        # iteration over network layers
        for layer_idx, layer in enumerate(nn_architecture, 1):
            params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
            params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        return params_values

    def train(self, X, nn_architecture, epochs, learning_rate, sample_size, loss_function):
        # initiation of neural net parameters
        params_values = self.init_layers(nn_architecture, 2)
        # initiation of lists storing the history
        # of metrics calculated during the learning process
        total_loss = []
        final_loss = []

        # performing calculations for subsequent iterations
        for i in range(epochs):
            total_cost = 0
            print("EPOCH ", i)
            for index, sample in enumerate(X):
                if index >= sample_size:
                    break
                sample = sample.reshape(784, 1)
                # step forward
                Y_hat, cashe = self.full_forward_propagation(sample, params_values, nn_architecture)

                # print(np.linalg.norm(sample - Y_hat))
                total_cost += np.linalg.norm(sample - Y_hat)

                # step backward - calculating gradient
                grads_values = self.full_backward_propagation(Y_hat, sample, cashe, params_values, nn_architecture,
                                                              loss_function)
                # updating model state
                params_values = self.update(params_values, grads_values, nn_architecture, learning_rate)

            total_loss.append(total_cost)
            print("Loss: ", total_cost)
            print("-----------")
            final_loss = total_cost
        return params_values, total_loss, final_loss

    def save(self, params, name):
        f = open(name + ".txt", "w")

        for key in params.keys():
            f.write(name + "-" + key + "\n")
            np.save(name + "-" + key, params[key])
        f.close()

    def load(self, file_path):
        params = {}
        f = open(file_path, 'r')
        for key in f:
            key = key.rstrip()
            param_key = key.split("-")[1]
            params[param_key] = np.load(key + ".npy")

        return params


class Functions:
    def __sub__(self, other):
        return

    # Activation functions
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def relu(Z):
        return Z * (Z > 0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    # Derrivatives of activation functions
    @staticmethod
    def sigmoid_prime(dA_curr, Z):
        sig = Functions.sigmoid(Z)
        return dA_curr * sig * (1 - sig)

    @staticmethod
    def relu_prime(dA_curr, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return dA_curr * x

    @staticmethod
    def tanh_prime(dA_curr, x):
        return dA_curr * (1 - np.tanh(x) ** 2)

    # Loss functions
    @staticmethod
    def cross_entropy(Y_hat, Y):
        # number of examples
        m = Y_hat.shape[1]
        # calculation of the cost according to the formula
        cost = -(1.0 / m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return cost

    @staticmethod
    def cross_entropy_prime(yHat, Y):
        return - (np.divide(Y, yHat) - np.divide(1 - Y, 1 - yHat))


    @staticmethod
    def euclidean_distance(yHat, y):
        return (y - yHat) ** 2

    @staticmethod
    def euclidean_distance_prime(yHat, y):
        return -2 * (y - yHat)
