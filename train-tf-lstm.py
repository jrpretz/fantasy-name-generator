import tensorflow as tf
import numpy as np
from util import encode_training_data
import sys
import random
import h5py

# "Wc"
Wca = tf.Variable(np.random.randn(50,50))
Wcx = tf.Variable(np.random.randn(50,29))
bc = tf.Variable(np.random.randn(50,1))

# "Wu"
Wua = tf.Variable(np.random.randn(50,50))
Wux = tf.Variable(np.random.randn(50,29))
bu = tf.Variable(np.random.randn(50,1))

# "Wf"
Wfa = tf.Variable(np.random.randn(50,50))
Wfx = tf.Variable(np.random.randn(50,29))
bf = tf.Variable(np.random.randn(50,1))

# "Wo"
Woa = tf.Variable(np.random.randn(50,50))
Wox = tf.Variable(np.random.randn(50,29))
bo = tf.Variable(np.random.randn(50,1))

# "Wy"
Wya = tf.Variable(np.random.randn(29,50))
by = tf.Variable(np.random.randn(29,1))


infile = open("names.txt")

max_depth = 20

As = []
C_alts = []
Cs = []
Ys = []
Gus = []
Gfs = []
Gos = []
Ypreds = []
costs = []
individual_costs = []

trainers = []
gradients = []
a_seed = tf.constant(np.zeros(shape=(50,1)))
x_seed = tf.constant(np.zeros(shape=(29,1)))
c_seed = tf.constant(np.zeros(shape=(50,1)))
#a_seed = tf.Variable(np.random.randn(50,1))
#x_seed = tf.Variable(np.random.randn(29,1))
#c_seed = tf.Variable(np.random.randn(50,1))

for i in range(0,max_depth):
    Ys.append(tf.placeholder(shape=(29,1),dtype=tf.float64,name="x%02d"%(i)))

    x_prev = None
    a_prev = None
    c_prev = None
    if(i == 0):
        x_prev = x_seed
        a_prev = a_seed
        c_prev = c_seed
    else:
        x_prev = Ys[i-1]
        a_prev = As[i-1]
        c_prev = Cs[i-1]

    c_alt = tf.nn.tanh(tf.matmul(Wca,a_prev) + tf.matmul(Wcx,x_prev)+bc)
    gu = tf.sigmoid(tf.matmul(Wua,a_prev) + tf.matmul(Wux,x_prev) + bu)
    gf = tf.sigmoid(tf.matmul(Wfa,a_prev) + tf.matmul(Wfx,x_prev) + bf)
    go = tf.sigmoid(tf.matmul(Woa,a_prev) + tf.matmul(Wox,x_prev) + bo)
    c = tf.add(tf.multiply(gu,c_alt),tf.multiply(gf,c_prev))
    a = tf.multiply(go,tf.nn.tanh(c))
    y = tf.nn.softmax(tf.matmul(Wya,a)+by,dim=0)

    cost = -tf.reduce_sum(Ys[i] * tf.log(y))
    individual_costs.append(cost)

    if(i == 0):
        costs.append(cost)
    else:
        costs.append(costs[i-1] +cost) 

    
    Cs.append(c)
    C_alts.append(c_alt)
    As.append(a)
    Ypreds.append(y)
        
    train_op = tf.train.MomentumOptimizer(0.0001,momentum=0.9).minimize(costs[i])
    #train_op = tf.train.AdamOptimizer(0.001).minimize(costs[i])
    trainers.append(train_op)
    #grad = tf.gradients(costs[i],[Waa,Wax,Wya,b,by])
    #gradients.append(grad)

index_to_char,char_to_index,encoded_words = encode_training_data("names.txt")


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(0,100000):
        
        selected_word = random.randint(0,len(encoded_words)-1)

        word = encoded_words[selected_word]

        feed_dict = {}
        for i in range(0,len(word)):
            y = np.zeros(shape=(29,1))
            y[word[i]] = 1.0

            feed_dict[Ys[i]] = y


        update,cost = session.run([trainers[len(word)-1],costs[len(word)-1]],feed_dict=feed_dict)
        print(epoch,cost/len(word))

        #for i in range(0,len(word)):
        #    print(session.run(Tests[i],feed_dict=feed_dict).T)
            
        #    print("-------------")
        #print(session.run(costs[len(word)-1],feed_dict=feed_dict))


    outfile = h5py.File("trained-weights-lstm.hd5","w")

    outfile.create_dataset("Wca",data=session.run(Wca))
    outfile.create_dataset("Wcx",data=session.run(Wcx))
    outfile.create_dataset("bc",data=session.run(bc))

    outfile.create_dataset("Wua",data=session.run(Wua))
    outfile.create_dataset("Wux",data=session.run(Wux))
    outfile.create_dataset("bu",data=session.run(bu))

    outfile.create_dataset("Wfa",data=session.run(Wfa))
    outfile.create_dataset("Wfx",data=session.run(Wfx))
    outfile.create_dataset("bf",data=session.run(bf))

    outfile.create_dataset("Woa",data=session.run(Woa))
    outfile.create_dataset("Wox",data=session.run(Wox))
    outfile.create_dataset("bo",data=session.run(bo))

    outfile.create_dataset("Wya",data=session.run(Wya))
    outfile.create_dataset("by",data=session.run(by))

    charmap = []
    for i in range(0,len(index_to_char)):
        charmap.append(index_to_char[i])
    outfile.create_dataset("ix_to_char",data=np.array(charmap,dtype='S'))
