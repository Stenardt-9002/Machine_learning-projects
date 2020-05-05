import nltk 
from nltk.stem.lancaster import LancasterStemmer
import numpy 
import tflearn
import tensorflow
import os
import random 
import json 



stemmer = LancasterStemmer()

with open("data1.json") as file:
    data = json.load(file)

# print(data)

words_lot = []
labels_lot = []
docs_lot = []

for ine1 in data["intents"]:
    for patter in ine1["patterns"]:
        #stemming
        word_list = nltk.word_tokenize(patter)
        words_lot.extend(word_list)

        pass



