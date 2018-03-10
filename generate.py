import h5py
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def read_network(infile):
    weights = h5py.File(infile)
    ix_to_char = weights["ix_to_char"]
    char_to_ix = {}
    for i in range(0,len(ix_to_char)):
        char_to_ix[ix_to_char[i].decode('ascii')] = i
    ix_to_char = {}
    for char,ix in char_to_ix.items():
        ix_to_char[ix] = char

    network = {}
    for paramName in ["Waa","Wax","Wya","b","by"]:
        param = weights[paramName]
        network[paramName] = np.array(param)
    network["ix_to_char"] = ix_to_char
    network["char_to_ix"] = char_to_ix

    return network

def sample_forward(network,a_prev,x_prev,depth):
    if depth > 50:
        return ""

    Wax = network["Wax"]
    Wya = network["Wya"]
    Waa = network["Waa"]
    b = network["b"]
    by = network["by"]

    a = np.tanh(np.matmul(Wax,x_prev) + np.matmul(Waa,a_prev) + b)
    y = softmax(np.matmul(Wya,a) + by)
    idx = np.random.choice(list(range(y.shape[0])), p=y.ravel())

    x = np.zeros_like(x_prev)
    x[idx] = 1.0

    char = network["ix_to_char"][idx]
    if(char == '\n'):
        return ""
    else:
        return char + sample_forward(network,a,x,depth+1)

def train_forward(network,a_prev,x,y,cache,depth):

    Wax = network["Wax"]
    Wya = network["Wya"]
    Waa = network["Waa"]
    b = network["b"]
    by = network["by"]

    a = np.tanh(np.matmul(Wax,x_prev) + np.matmul(Waa,a_prev) + b)
    y_pred = softmax(np.matmul(Wya,a) + by)



network = read_network("trained-weights-test.h5")


a_prev = np.zeros(shape=(50,1))
x_prev = np.zeros(shape=(29,1))

for i in range(0,100):
    s = sample_forward(network,a_prev,x_prev,0)
    s = s[0].upper() + s[1:]
    print(s)
