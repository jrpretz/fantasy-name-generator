import numpy as np
import tensorflow as tf
import h5py


def encode_training_data(inputFile):
    infile = open(inputFile)
    characters = set()
    characters.add('\n')
    words = []
    for line in infile:
        line = line.rstrip().lstrip().lower()
        for i in range(len(line)):
            characters.add(line[i])
        words.append(line)
    characters = list(characters)
    characters.sort()
    char_to_index = {}
    index_to_char = {}
    for i in range(0,len(characters)):
        index_to_char[i] = characters[i]
        char_to_index[characters[i]] = i
    nCharacters = len(characters)
    encoded_words = []
    for word in words:
        encoded = np.zeros(shape=(len(word)+1),dtype=np.int)
        for j in range(len(word)):
            encoded[j] = char_to_index[word[j]]
        encoded[len(word)] = char_to_index['\n']
        encoded_words.append(encoded)

    return index_to_char,char_to_index,encoded_words

def create_network(parameters):
    network = Object()
    
    network.Waa = Waa = tf.Variable(parameters.Waa)
    network.Wax = Waa = tf.Variable(parameters.Wax)
    network.Wya = Wya = tf.Variable(parameters.Wya)
    network.by = by = tf.Variable(parameters.by)
    network.b = b = tf.Variable(parameters.b)

    network.Waa_assign = tf.placeholder(shape=Waa.shape)
    network.Wax_assign = tf.placeholder(shape=Wax.shape)
    network.Wya_assign = tf.placeholder(shape=Wya.shape)
    network.by_assign = tf.placeholder(shape=by.shape)
    network.b_assign = tf.placeholder(shape=b.shape)

    network.update_Waa = Waa.assign(network.Waa_assign)
    network.update_Wax = Wax.assign(network.Wax_assign)
    network.update_Wya = Wya.assign(network.Wya_assign)
    network.update_by = by.assign(network.by_assign)
    network.update_b = b.assign(network.b_assign)

    n_a,n_x = n_a,n_x = Wax.shape
    n_y,_ = Wya.shape

    network.n_a = n_a
    network.n_x = n_x
    network.n_y = n_y
    
    network.a_input = a_input = tf.placeholder(dtype=tf.float64,shape=(n_a,1))
    network.x_input = x_input = tf.placeholder(dtype=tf.float64,shape=(n_x,1))
    network.y = y = tf.placeholder(dtype=tf.float64,shape=(n_y,1))
    
    network.a_output = a_output = tf.tanh(tf.matmul(Waa,a_input)+tf.matmul(Wax,x_input)+b)
    network.y_output = y_output = tf.nn.softmax(tf.matmul(Wya,a_output)+by,dim=0)

    network.cost = tf.reduce_sum(y * tf.log(y_output))
    
    network.gradients = tf.gradients(network.cost,[Waa,Wax,Wya,by,b])
    
    return network
