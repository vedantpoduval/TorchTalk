'''Preprocessiing workflow
    Tokenize ---> Stemming(lower case) ---> Bag of words'''
import nltk
from nltk.stem.porter import PorterStemmer
Stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return Stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence,all_words):
    pass
sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good?"""
words = ["Organize","organ","organizes"]
print(tokenize(sentence))
print([stem(w) for w in words])