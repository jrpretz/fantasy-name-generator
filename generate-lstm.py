import h5py
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1./(1+np.exp(-x))

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
    
    for paramName in ["Wca","Wcx","bc",
                      "Wua","Wux","bu",
                      "Wfa","Wfx","bf",
                      "Woa","Wox","bo",
                      "Wya","by"]:
        param = weights[paramName]
        network[paramName] = np.array(param)
    network["ix_to_char"] = ix_to_char
    network["char_to_ix"] = char_to_ix

    return network

def sample_forward(network,a_prev,x_prev,c_prev,depth):
    if depth > 50:
        return ""

    Wca = network["Wca"]
    Wcx = network["Wcx"]
    bc = network["bc"]

    Wua = network["Wua"]
    Wux = network["Wux"]
    bu = network["bu"]

    Wfa = network["Wfa"]
    Wfx = network["Wfx"]
    bf = network["bf"]

    Woa = network["Woa"]
    Wox = network["Wox"]
    bo = network["bo"]

    Wya = network["Wya"]
    by = network["by"]

    c_alt = np.tanh(np.matmul(Wca,a_prev) + np.matmul(Wcx,x_prev)+bc)
    gu = sigmoid(np.matmul(Wua,a_prev) + np.matmul(Wux,x_prev) + bu)
    gf = sigmoid(np.matmul(Wfa,a_prev) + np.matmul(Wfx,x_prev) + bf)
    go = sigmoid(np.matmul(Woa,a_prev) + np.matmul(Wox,x_prev) + bo)
    c = np.add(np.multiply(gu,c_alt),np.multiply(gf,c_prev))
    a = np.multiply(go,np.tanh(c))
    y = softmax(np.matmul(Wya,a)+by)

    idx = np.random.choice(list(range(y.shape[0])), p=y.ravel())

    x = np.zeros_like(x_prev)
    x[idx] = 1.0

    char = network["ix_to_char"][idx]
    if(char == '\n'):
        return ""
    else:
        return char + sample_forward(network,a,x,c,depth+1)



network = read_network("trained-weights-lstm.hd5")


a_prev = np.zeros(shape=(50,1))
c_prev = np.zeros(shape=(50,1))
x_prev = np.zeros(shape=(29,1))

for i in range(0,100):
    s = sample_forward(network,a_prev,x_prev,c_prev,0)
    if len(s) > 1:
        s = s[0].upper() + s[1:]
        print(s)
