import tensorflow as tf
import numpy as np
from util import encode_training_data
import sys
import random
import h5py

Waa = tf.Variable(np.random.randn(50,50))
Wax = tf.Variable(np.random.randn(50,29))
Wya = tf.Variable(np.random.randn(29,50))
b = tf.Variable(np.random.randn(50,1))
by = tf.Variable(np.random.randn(29,1))

infile = open("names.txt")

max_depth = 20

Xs = []
Ys = []
Ypreds = []
As = []
Zs = []
costs = []

trainers = []

#a_seed = tf.Variable(np.zeros(shape=(50,1)))
#a_seed = tf.Variable(np.random.randn(50,1))
#a_seed = tf.Variable(np.random.randn(50,1))
a_seed = tf.constant(np.zeros(shape=(50,1)))

for i in range(0,max_depth):
    Xs.append(tf.placeholder(shape=(29,1),dtype=tf.float64,name="x%02d"%(i)))

    if(i==0):
        a = tf.tanh(tf.matmul(Waa,a_seed)+tf.matmul(Wax,Xs[i]) + b)
    else:
        a = tf.tanh(tf.matmul(Waa,As[i-1])+tf.matmul(Wax,Xs[i]) + b)
    As.append(a)

    z = tf.matmul(Wya,a) + by
    Zs.append(z)

    Ys.append(tf.placeholder(shape=(29,1),dtype=tf.float64))

    y = tf.nn.softmax(z,dim=0)

    Ypreds.append(y)

    cost = -tf.reduce_sum(Ys[i] * tf.log(y))

    if(i == 0):
        costs.append(cost)
    else:
        costs.append(costs[i-1] +cost) 

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(costs[i])
    #train_op = tf.train.AdamOptimizer(0.001).minimize(costs[i])
    trainers.append(train_op)

index_to_char,char_to_index,encoded_words = encode_training_data("names.txt")


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(0,100000):
        selected_word = random.randint(0,len(encoded_words)-1)

        word = encoded_words[selected_word]

        feed_dict = {}
        for i in range(0,len(word)-1):
            x = np.zeros(shape=(29,1))
            x[word[i]] = 1.0

            y = np.zeros(shape=(29,1))
            y[word[i+1]] = 1.0

            feed_dict[Xs[i]] = x
            feed_dict[Ys[i]] = y


        update,cost = session.run([trainers[len(word)-2],costs[len(word)-2]],feed_dict=feed_dict)
        print(epoch,cost/(len(word)-1))


    outfile = h5py.File("trained-weights-test.h5","w")
    outfile.create_dataset("Waa",data=session.run(Waa))
    outfile.create_dataset("Wax",data=session.run(Wax))
    outfile.create_dataset("Wya",data=session.run(Wya))
    outfile.create_dataset("b",data=session.run(b))
    outfile.create_dataset("by",data=session.run(by))
    charmap = []
    for i in range(0,len(index_to_char)):
        charmap.append(index_to_char[i])
    outfile.create_dataset("ix_to_char",data=np.array(charmap,dtype='S'))
