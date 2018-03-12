# fantasy-name-generator
Simple Recurrent Neural Network (RNN) to generate fantasy character names

The training set is from: 
http://www.skatoolaki.com/eq/n_fantasynamegenerator.htm

It is way beyond my scope to explain what an RNN is. But perchance somebody
ever wants to look at or use this code, I figured a word of explanation
about how I implemented it was in order. 

The main guts for the network are in train-tf-lstm.py. It is a Long
Short Term Memory RNN unit. It takes in the prior activations of the network,
a, the prior letter in the word (encoded as a one-hot vector), x,
and the state of the memory cell, c. It spits out a probability distribution
for the next letter as well as the memory cell, c, and the internal activations,
a.

The trick in these things is to train them. Since I am training a network for
learning names, it has a fairly restricted length, no more than 10 or so
characters in a word is typical. So I trained by just creating N "unrolled"
networks. Network 1 is one RNN cell. Network 2 is 2 RNN cells and so on.
In training I use whatever network is appropriate for the word I am training
on.

The script generate-lstm.py will generate names for you using the trained
weights. The hdf5 file trained-weights-lstm.hd5 has already been trained. You
can train it again using the script train-tf-lstm.py.

The code is not going to be winning any awards, but it functions and is
clear how it works.
