import nltk 
from nltk.stem.lancaster import LancasterStemmer
import numpy as np 
import tflearn
import tensorflow
import os
import random 
import json 
import pickle 


stemmer = LancasterStemmer()

with open("data1.json") as file:
    data = json.load(file)

# print(data)

# try:
#     with open("Bkupsave","rb") as f:
#         words_lot,labels_lot,training_1,out_put = pickle.load(f)
# except :
#     pass



words_lot = []
labels_lot = []
doc_x = []
doc_y = []
docs_lot = []

for ine1 in data["intents"]:
    for patter in ine1["patterns"]:
        #stemming
        word_list = nltk.word_tokenize(patter)
        words_lot.extend(word_list)
        # docs_lot.append(patter)
        doc_x.append(word_list)
        doc_y.append(ine1["tag"])
    if ine1["tag"] not in labels_lot:
        labels_lot.append(ine1["tag"])



# print(doc_x)      
# print("\n\n\n\n")

# print("words_lot = ",words_lot)
# print("\n\n\n\n")

# print("doc_x = ",doc_x)
# print("\n\n\n\n")
# print("doc_y = ",doc_y)
# print("\n\n\n\n")

# print("labels_lot = ",labels_lot)
words = [stemmer.stem(w.lower()) for w in words_lot if w != "?"]
# print(words)
# print("\n\n\n\n")

words = sorted(list(set(words)))
# print("\n\n\n\n")
# print("words = ",words)
# print(words)
# print("\n\n\n\n")


labelsnew = sorted(labels_lot)
# print("labelsnew = ",labelsnew)
training_1 = []
out_put = []


out_put_empty = [0 for _ in range(len(labelsnew))]


for x,doc in enumerate(doc_x):
    bag = []

    wrds = [stemmer.stem(wc) for wc in doc ]

    for wc in words:
        if wc in wrds:
            bag.append(1)

        else:
            bag.append(0)

    output_trow = out_put_empty[:]
    output_trow[labelsnew.index(doc_y[x])] = 1        

    training_1.append(bag)
    out_put.append(output_trow)


training_1 = np.array(training_1)
out_put = np.array(out_put)



tensorflow.reset_default_graph()

net1 = tflearn.input_data(shape=[None,len(training_1[0])])

net1 = tflearn.fully_connected(net1,8)
net1 = tflearn.fully_connected(net1,8)
net1 = tflearn.fully_connected(net1,len(out_put[0]),activation="softmax")
net  = tflearn.regression(net1)

mainmpdel = tflearn.DNN(net)

# try:
#     mainmpdel.load("Bkupsave")

# except:
#     mainmpdel.fit(training_1,out_put,n_epoch=1000,batch_size=8,show_metric=True)
#     mainmpdel.save("Bkupsave")
mainmpdel.fit(training_1,out_put,n_epoch=1000,batch_size=8,show_metric=True)
mainmpdel.save("Bkupsave")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(wc.lower()) for wc in s_words]

    for sec in s_words:
        for i,w in enumerate(words):
            if w==sec:
                bag[i] = 1
                pass
    return np.array(bag)



def chat():
    print("INitaite Sequence The bot (quit to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
            pass

        resylt = mainmpdel.predict([bag_of_words(inp,words)])
        # print(resylt)
        # get max prob 
        main_resylt_idex = np.argmax(resylt)
        tag = labels_lot[main_resylt_idex]
        # response = 
        for tag1 in data["intents"]:
            if tag1["tag"] == tag:
                respomse = tag1["responses"]
        print(respomse)

        print(random.choice(respomse))



print(out_put)
 
chat()

