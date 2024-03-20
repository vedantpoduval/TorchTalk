'''Preprocessiing workflow
    Tokenize ---> Stemming(lower case) ---> Bag of words'''
import nltk
import torch
import numpy as np
from nltk.stem.porter import PorterStemmer
Stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return Stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = torch.zeros(len(all_words))
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag