import tensorflow as tf
import numpy as np
from util import encode_training_data
import sys
import random
import h5py

alpha = 0.01

Waa = tf.Variable(np.random.randn(50,50))
Wax = tf.Variable(np.random.randn(50,29))
Wya = tf.Variable(np.random.randn(29,50))
b = tf.Variable(np.random.randn(50,1))
by = tf.Variable(np.random.randn(29,1))

dWaa = tf.placeholder(shape=(50,50),dtype=tf.float64)
dWax = tf.placeholder(shape=(50,29),dtype=tf.float64)
dWya = tf.placeholder(shape=(29,50),dtype=tf.float64)
db = tf.placeholder(shape=(50,1),dtype=tf.float64)
dby = tf.placeholder(shape=(29,1),dtype=tf.float64)

update_Waa = tf.assign(Waa,Waa - alpha * dWaa)
update_Wax = tf.assign(Wax,Wax - alpha * dWax)
update_Wya = tf.assign(Wya,Wya - alpha * dWya)
update_b = tf.assign(b,b - alpha * db)
update_by = tf.assign(by,by - alpha * dby)

infile = open("names.txt")

max_depth = 20

#Xs = []
Ys = []
Ypreds = []
As = []
Zs = []
costs = []

trainers = []
gradients = []
#a_seed = tf.Variable(np.zeros(shape=(50,1)))
#a_seed = tf.Variable(np.random.randn(50,1))
#a_seed = tf.Variable(np.random.randn(50,1))
a_seed = tf.constant(np.zeros(shape=(50,1)))
x_seed = tf.constant(np.zeros(shape=(29,1)))

for i in range(0,max_depth):
    Ys.append(tf.placeholder(shape=(29,1),dtype=tf.float64,name="x%02d"%(i)))

    if(i==0):
        a = tf.nn.tanh(tf.matmul(Waa,a_seed)+tf.matmul(Wax,x_seed) + b)
    else:
        a = tf.nn.tanh(tf.matmul(Waa,As[i-1])+tf.matmul(Wax,Ys[i-1]) + b)
    As.append(a)

    z = tf.matmul(Wya,a) + by
    Zs.append(z)

    y = tf.nn.softmax(z,dim=0)

    Ypreds.append(y)

    cost = -tf.reduce_sum(Ys[i] * tf.log(y))

    if(i == 0):
        costs.append(cost)
    else:
        costs.append(costs[i-1] +cost) 

    train_op = tf.train.MomentumOptimizer(0.001,momentum=0.9).minimize(costs[i])
    #train_op = tf.train.AdamOptimizer(0.001).minimize(costs[i])
    trainers.append(train_op)
    grad = tf.gradients(costs[i],[Waa,Wax,Wya,b,by])
    gradients.append(grad)

index_to_char,char_to_index,encoded_words = encode_training_data("names.txt")

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(0,10000):

        dWaa_accumulate = np.zeros(shape=dWaa.shape)
        dWax_accumulate = np.zeros(shape=dWax.shape)
        dWya_accumulate = np.zeros(shape=dWya.shape)
        db_accumulate = np.zeros(shape=db.shape)
        dby_accumulate = np.zeros(shape=dby.shape)

        nBatches = 10

        avgCost = 0.
        
        for batch in range(0,nBatches):
        
            selected_word = random.randint(0,len(encoded_words)-1)

            word = encoded_words[selected_word]

            feed_dict = {}
            for i in range(0,len(word)):
                y = np.zeros(shape=(29,1))
                y[word[i]] = 1.0

                feed_dict[Ys[i]] = y


            cost,grads = session.run([costs[len(word)-1],gradients[len(word)-1]],feed_dict=feed_dict)
            #update,cost,grads = session.run([trainers[len(word)-1],costs[len(word)-1],gradients[len(word)-1]],feed_dict=feed_dict)
            
            avgCost += cost/(len(word))

            for i in range(0,5):
                np.clip(grads[i],-5,5,grads[i])
            
            dWaa_accumulate += grads[0]
            dWax_accumulate += grads[1]
            dWya_accumulate += grads[2]
            db_accumulate += grads[3]
            dby_accumulate += grads[4]

        print(epoch,avgCost/nBatches)


        
        dWaa_accumulate /= nBatches
        dWax_accumulate /= nBatches
        dWya_accumulate /= nBatches
        db_accumulate /= nBatches
        dby_accumulate /= nBatches

        session.run(update_Waa,feed_dict={dWaa:dWaa_accumulate})
        session.run(update_Wax,feed_dict={dWax:dWax_accumulate})
        session.run(update_Wya,feed_dict={dWya:dWya_accumulate})
        session.run(update_b,feed_dict={db:db_accumulate})
        session.run(update_by,feed_dict={dby:dby_accumulate})

        
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
