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
