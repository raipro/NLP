#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sys
path = sys.argv[1]
f = open("../intro-to-nlp-assign3/europarl-corpus/train.europarl","r+")
import re
import random
import string
import tensorflow as tf
import tflearn
import numpy as np
from keras.layers import Embedding,LSTM,Dense
from keras.models import Sequential
data = []
dat = f.readlines()
for it1 in dat:
    it = it1
    it = re.sub("[\.]+"," \. ",it)
    it = re.sub("[>]+",">",it)
    it = re.sub("[<]+","<",it)
    it = re.sub("[\,]+"," \, ",it)
    it = re.sub("[\']+"," \' ",it)
    it = re.sub("[\"]+"," \" ",it)
    it = re.sub("[\;]+"," \; ",it)
    it = re.sub("[\:]+"," \: ",it)
    it = re.sub("[!]+"," ! ",it)
    it = re.sub("[\?]+"," \? ",it)
    it = re.sub("[\/]+"," \/ ",it)
    # re.sub("[\s]+","\s",it)
    # f.write(it)
    it = it.split()
    data.append(it)

voc = set([])
str2 = 'PAD'
voc.add(str2)
for it in data:
    for it1 in it:
        if it1 not in voc:
            voc.add(it1)


gg = dict()
gg1 = dict()
i = 0
for x in voc:
    gg[i] = x
    gg1[x] = i
    i += 1

if gg1[str2]!=0:
    d1 = gg1[str2]
    str1 = gg[0]
    gg1[str1] = d1
    gg[0] = str2
    gg[d1] = str1
    gg1[str2] = 0

# print(gg1['PAD'])
# print(gg[0])

             
# #                 cury = t_categorical(cury,len(voc))
            
                
                
            
                
            
    


# # In[22]:


# print(len(voc))
# print(len(gg1))


# # In[100]:


# def train(lr = 0.01,n = 5,batch_size = 5000,ep = 10,hidden = 256):
#     context = []
#     target = []
#     for it in data:
#         it1 = ['PAD','PAD','PAD','PAD']
#         it1 = it1 + it
#         for j in range(0,len(it1) - n + 1):
#             temp = []
#             for j1 in range(0,n-1):
#                 temp.append(gg1[it1[j+j1]])
#             context.append(temp)
#             target.append(gg1[it1[j+n-1]])
#     context = np.array(context)
#     target = np.array(target)
#     print(context.shape)
#     print(target.shape)
# #     return
# #     w2v = np.random.rand(len(gg1), 30) 
# #     EMBEDDING_SIZE = w2v.shape[-1]
# #     embedding_matrix = tf.constant(w2v, dtype=tf.float32)
# #     net = tflearn.input_data([None, n-1], dtype=tf.int32, name='input')
# #     net = tflearn.embedding(net, input_dim=len(voc), output_dim=EMBEDDING_SIZE,
# #                         weights_init=embedding_matrix, trainable=True)
# #     net = tflearn.lstm(net, hidden, dynamic=True)
# #     net = tflearn.fully_connected(net, len(voc), activation='softmax')
# #     net = tflearn.regression(net, optimizer='adam', learning_rate=lr,
# #                          loss='categorical_crossentropy', name='target')
# #     model = tflearn.DNN(net)
#     model = Sequential()
#     model.add(Embedding(input_dim = len(voc),output_dim = 25,input_length = 4))
#     model.add(LSTM(18))
#     model.add(Dense(len(voc),activation = 'softmax'))
#     model.compile('adam','categorical_crossentropy')
#     d = 0;
#     while d < len(context):
#         print(d/batch_size)
#         curx = context[d:min(len(context),d+batch_size)]
#         cury = target[d:min(len(context),d+batch_size)]
# #         print(curx)
# #         print(cury)
#         d += batch_size
#         curyt = []
#         for tot in cury:
#             cury1 = np.zeros(len(voc))
#             cury1[tot] = 1
#             curyt.append(cury1)
#             if tot>=len(voc):
#                 print('fck')
#         curx = np.array(curx)
#         curyt = np.array(curyt)
#         print(curx.shape)
#         print(curyt.shape)
#         model.fit(curx,curyt,epochs = 10)
# #         model.save("./trained_model")
#     return model
    


# In[101]:


# model = train()


# In[103]:


# model.save("./trained_model")


# In[104]:


model = tf.keras.models.load_model(path)


# In[105]:


# import random
# f2 = open("2019101096_LM_train.txt", "a")
# def train_perplexity_scores():
#     avg = 0
#     j3 = 0
#     for it in data:
#         j3 += 1
#         print(j3)
# #         print(it)
#         if len(it) == 0:
# #             f2.write("Sentence_" + str(j3)  + ": " + '1' + '\n')
#             avg += 1
#             continue
#         it1 = ['PAD','PAD','PAD','PAD']
#         it1 = it1 + it
#         gg2 = []
#         for j in range(4,len(it1)):
#             temp = []
#             for j1 in range(0,4):
# #                 print(it1[j+j1-3])
#                 temp.append(gg1[it1[j+j1-4]])
# #             temp.reverse()
# #             print(temp)
#             gg2.append(temp) 
#         gg2 = np.array(gg2)
# #         print(gg2.shape)
#         probs = model.predict(gg2)
# #         print(probs)
#         prob2 = 1
#         for it2 in range(0,len(probs)):
#             prob2 *= probs[it2][gg1[it1[it2+4]]]
#         if prob2 < pow(10,-300):
#             prob2 = random.uniform(0.00001,0.00001)
#         score = pow(1/prob2,1/len(it1))
# #         f2.write("Sentence_" + str(j3)  + ": " + str(score) + '\n')
#         avg += score
#     avg = avg/len(data)
#     print(avg)
#     f2.write("avg_Perplexity:" + str(avg))   
# train_perplexity_scores()           


# In[81]:


# f1 = open("./intro-to-nlp-assign3/europarl-corpus/test.europarl","r+")
# dat = f1.readlines();
# data1 = []
# for it1 in dat:
#     it = it1
#     it = re.sub("[\.]+"," \. ",it)
#     it = re.sub("[>]+",">",it)
#     it = re.sub("[<]+","<",it)
#     it = re.sub("[\,]+"," \, ",it)
#     it = re.sub("[\']+"," \' ",it)
#     it = re.sub("[\"]+"," \" ",it)
#     it = re.sub("[\;]+"," \; ",it)
#     it = re.sub("[\:]+"," \: ",it)
#     it = re.sub("[!]+"," ! ",it)
#     it = re.sub("[\?]+"," \? ",it)
#     it = re.sub("[\/]+"," \/ ",it)
#     # re.sub("[\s]+","\s",it)
#     # f.write(it)
#     it = it.split()
#     data1.append(it)




        
       
        


# In[94]:


# import random
# f3 = open("2019101096_LM_test.txt", "a")
# def test_perplexity_scores():
#     avg = 0
#     j3 = 0
#     for it in data1:
#         j3 += 1
# #         print(j3)
# #         print(it)
#         if len(it) == 0:
#             print("Sentence_" + str(j3)  + ": " + '1' + '\n')
#             avg += 1
#             continue
#         it1 = ['PAD','PAD','PAD','PAD']
#         it1 = it1 + it
#         gg2 = []
#         flag = 0
#         for j in range(4,len(it1)):
#             temp = []
#             for j1 in range(0,4):
# #                 print(it1[j+j1-3])
#                 if it1[j+j1-4] not in gg1:
#                     k1 = random.randint(1,len(voc))
#                     it1[j+j1-4] = gg[k1]
#                     gg1[gg[k1]] = k1
#                     temp.append(k1)
#                 else:
#                     temp.append(gg1[it1[j+j1-4]])
# #             temp.reverse()
# #             print(temp)
#             gg2.append(temp)
#         if flag == 0:
#             gg2 = np.array(gg2)
#     #         print(gg2.shape)
#             probs = model.predict(gg2)
#     #         print(probs)
#             prob2 = 1
#             for it2 in range(0,len(probs)):
#                 prob2 *= probs[it2][gg1[it1[it2+4]]]
#             if prob2 < pow(10,-300):
#                 prob2 = random.uniform(0.00001,0.00001)
#             score = pow(1/prob2,1/len(it1))
#             print("Sentence_" + str(j3)  + ": " + str(score) + '\n')
#             avg += score
#     avg = avg/len(data1)
#     print("avg_Perplexity:" + str(avg))  
# test_perplexity_scores()


# In[106]:


sentence = input("Enter sentence: ")
sentence = sentence.split()
if len(sentence) == 0:
    print(0)
else:
    it1 = ['PAD','PAD','PAD','PAD']
    it1 = it1 + sentence
    gg2 = []
    flag = 0
    for j in range(4,len(it1)):
        temp = []
        for j1 in range(0,4):
    #                 print(it1[j+j1-3])
            if it1[j+j1-4] not in gg1:
                k1 = random.randint(1,len(voc))
                it1[j+j1-4] = gg[k1]
                gg1[gg[k1]] = k1
                temp.append(k1)
            else:
                temp.append(gg1[it1[j+j1-4]])
    #             temp.reverse()
    #             print(temp)
        gg2.append(temp)
    if flag == 0:
        gg2 = np.array(gg2)
    #         print(gg2.shape)
        probs = model.predict(gg2)
    #         print(probs)
        prob2 = 1
        for it2 in range(0,len(probs)):
            prob2 *= probs[it2][gg1[it1[it2+4]]]
        print(prob2)



