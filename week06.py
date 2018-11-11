import math
import operator
from functools import reduce
from random import random

import pandas
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

bias = 1
learning_rate = .1
neurons_per_layer = [2, 3]


def readInCar():
    car_data = r"C:\Users\casto\Documents\BYUI\Fall 2018\CS 450\carData.csv"
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    dataset = pandas.read_csv(car_data, names=headers, index_col=False)
    df = pandas.DataFrame(dataset)
    # df = dataset
    df = df.apply(LabelEncoder().fit_transform)

    # df = np.ndarray(df, df.shape)
    # scaled_data = pandas.get_dummies(df)
    # scaled_data = ((df - df.min()) / (df.max() - df.min()))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # np_scaled = min_max_scaler.fit_transform(df)
    # scaled_data = pandas.DataFrame(np_scaled)

    # scaler = StandardScaler()
    # scaler.fit(dataFrame)
    # scaled_data = scaler.transform(dataFrame)
    # col = len(df.columns)
    just_data = df.iloc[:, 1:]
    just_data = just_data.values
    targets = df.iloc[:, 0]
    targets = targets.values
    data_train, data_test, target_train, target_test = train_test_split(just_data, targets, test_size=0.30,
                                                                        random_state=42)

    data_train = np.matrix(data_train)
    data_test = np.matrix(data_test)
    # target_train = np.matrix(target_train)
    # target_test = np.matrix(target_test)
    return data_train, data_test, target_train, target_test


def morph_predictions(predictions):
    col = np.argmax(predictions, axis=0)
    # print(col)
    col = col.tolist()
    if len(col) == 1:
        col = col[0]
    # col = np.matrix(col)
    # col = np.matrix.flatten(col)
    # col = reduce(operator.concat, col)
    # print(col)
    return col


def transfer_derivative(output):
    return output * (1.0 - output)


def calc_error(predictions, targets):
    error = (targets - predictions) * predictions * (1.0 - predictions)
    return error


def backward(network, activations, targets):
    # finding errors
    # print('activations shape', activations.shape)
    errors = []
    # activations = np.matrix.transpose(activations)
    for j in range(0, len(activations)):
        activations[j] = np.matrix(activations[j])
        activations[j] = np.matrix.transpose(activations[j])
        activations[j] = np.squeeze(np.asarray(activations[j]))
    error = activations[-1] * (1 - activations[-1]) * (activations[-1] - targets)
    error = np.matrix(error)
    # print('error shape', error.shape)
    errors.append(error)
    last_error = error
    for i in reversed(range(0, len(network))):
        # for k in reversed(range(0, network[i].shape[0])):
        # print('last_error shape', last_error.shape)
        # print('network[i] shape', network[i].shape)
        error = (network[i] * last_error)
        errors.insert(0, error)
        last_error = error[0:-1:1]

    # update weights
    for m in range(0, len(network)):
        network[m] = network[m] - learning_rate * errors[m] * np.matrix.transpose(activations[m])

    return network


def add_bias_node(data):
    bias_tuple = data.shape
    new_tuple = tuple((bias_tuple[0], 1))
    # print(new_tuple)
    bias_array = np.full(new_tuple, bias)
    data = np.column_stack((bias_array, data))
    return data


def read_in_iris():
    iris = datasets.load_iris()

    scaler = StandardScaler()
    scaler.fit(iris.data)
    scaled_data = scaler.transform(iris.data)
    # add bias node
    data = add_bias_node(scaled_data)

    data_train, data_test, target_train, target_test = train_test_split(data, iris.target, test_size=0.30,
                                                          random_state = 42)

    data_train = np.matrix(data_train)
    data_test = np.matrix(data_test)
    print(type(target_train)) # = np.matrix(target_train)
    print(type(target_test)) # = np.matrix(target_test)
    return data_train, data_test, target_train, target_test


def set_up_network():
    # matrix of Neurons
    network = []
    data_num_2 = data_train.shape[1]
    for i in neurons_per_layer:
        layer = np.random.random((data_num_2, i))
        network.append(layer)
        data_num_2 = i + 1

    # print(network.shape)
    network_m = []
    for layer in network:
        layer_m = np.matrix(layer)
        network_m.append(layer_m)
    return network_m


def feed_forward(network, data):
    # compute predictions
    activations = []# np.ndarray((len(neurons_per_layer), len(data)))
    # print((len(neurons_per_layer), len(data)))
    data_working = data
    for i in range(0, len(network)):
        # print('data_working shape', data_working.shape)
        predictions = data_working * network[i]
        # predictions = np.matrix.transpose(predictions)
        # print('predictions shape', predictions.shape)
        predictions = 1 / (1 + np.exp(predictions * -1))
        # print('predictions shape', predictions.shape)
        # if i == 0:
        #     activations = predictions
        # else:
        #     activations = np.column_stack((activations, predictions))
        activations.append(predictions)
        # print('activations shape ff', activations.shape)
        if network[i] is not network[-1]:
            predictions = add_bias_node(predictions)
        data_working = predictions

    # print('activations shape ff', len(activations))
    # activations = np.matrix(activations)
    return activations


def calc_accuracy(predictions, targets):
    # print(predictions)
    # print(len(predictions))
    predictions = morph_predictions(predictions)
    # print(len(predictions))
    # print('targets', len(targets))
    # print('predictions', len(predictions))
    # find accuracy
    correct = 0
    for i in range(0, len(predictions)):
        if predictions[i] == targets[i]:
            correct += 1

    accuracy = correct / len(predictions)
    accuracy *= 100
    return accuracy
################################################################################################


# data_train, data_test, target_train, target_test = read_in_iris()
data_train, data_test, target_train, target_test = readInCar()

network = set_up_network()
# for layer in network:
#     print(layer.shape)
for i in range(0, 1000):
    activations = feed_forward(network, data_train)
    # print(activations[-1].shape)
    network = backward(network, activations, target_train)
    accuracy = calc_accuracy(activations[-1], target_train)
    print("{:.2f}%".format(accuracy))

print("###########################################################")
print("Testing:")
activations = feed_forward(network, data_test)
# np.ndarray(activations).T.tolist()
# activations = np.ndarray.transpose(activations)
# map(list,map(None,*activations))
activations = activations[-1]
# activations = activations[0]
activations = np.matrix.transpose(activations)

# activations = np.ndarray(activations)
# print(activations, activations.shape, type(activations))
accuracy = calc_accuracy(activations, target_test)
print("accuracy of neural network is {:.2f}%".format(accuracy))
# print(math.exp(1))
