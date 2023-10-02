#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
path = sys.argv[1]
# f = open("../intro-to-nlp-assign3/ted-talks-corpus/train.en","r+")
# f1 = open("../intro-to-nlp-assign3/ted-talks-corpus/train.fr","r+")
import re
import random
import string
import tensorflow as tf
import keras
# import tflearn
import numpy as np
from keras.layers import Embedding,LSTM,Dense
from keras.models import Sequential
# path = './Downloads/aad/'
# engd = []
# frd = []
# dat = f.readlines()
# dat1 = f1.readlines()
# en1 = 0
# fr1 = 0
# for it1 in dat:
#     it = it1
#     it = re.sub("[\.]+"," ",it)
#     it = re.sub("[>]+"," ",it)
#     it = re.sub("[<]+"," ",it)
#     it = re.sub("[\,]+"," ",it)
#     it = re.sub("[\']+"," ",it)
#     it = re.sub("[\"]+"," ",it)
#     it = re.sub("[\;]+"," ",it)
#     it = re.sub("[\:]+"," ",it)
#     it = re.sub("[!]+"," ",it)
#     it = re.sub("[\?]+"," ",it)
#     it = re.sub("[\/]+"," ",it)
#     # re.sub("[\s]+","\s",it)
#     # f.write(it)
#     it = it.split()
#     engd.append(it)
#     en1 = max(en1,len(it))

# for it1 in dat1:
#     it = it1
#     it = re.sub("[\.]+"," ",it)
#     it = re.sub("[>]+"," ",it)
#     it = re.sub("[<]+"," ",it)
#     it = re.sub("[\,]+"," ",it)
#     it = re.sub("[\']+"," ",it)
#     it = re.sub("[\"]+"," ",it)
#     it = re.sub("[\;]+"," ",it)
#     it = re.sub("[\:]+"," ",it)
#     it = re.sub("[!]+"," ",it)
#     it = re.sub("[\?]+"," ",it)
#     it = re.sub("[\/]+"," ",it)
#     # re.sub("[\s]+","\s",it)
#     # f.write(it)
#     it = it.split()
#     frd.append(it)
#     fr1 = max(fr1,len(it))

# voc = set([])
# str2 = '<PAD>'
# voc.add(str2)
# for it in engd:
#     for it1 in it:
#         if it1 not in voc:
#             voc.add(it1)

# voc1 = set([])
# str3 = '<pad>'
# voc1.add(str3)
# for it in frd:
#     for it1 in it:
#         if it1 not in voc1:
#             voc1.add(it1)

# gg = dict()
# gg1 = dict()
# i = 0
# for x in voc:
#     gg[i] = x;
#     gg1[x] = i;
#     i += 1

# if gg1[str2]!=0:
#     d1 = gg1[str2]
#     str1 = gg[0];
#     gg1[str1] = d1
#     gg[0] = str2
#     gg[d1] = str1
#     gg1[str2] = 0

# gg2 = dict()
# gg3 = dict()
# i = 0
# for x in voc1:
#     gg2[i] = x;
#     gg3[x] = i;
#     i += 1

# if gg3[str3]!=0:
#     d1 = gg3[str3]
#     str1 = gg2[0];
#     gg3[str1] = d1
#     gg2[0] = str2
#     gg2[d1] = str1
#     gg3[str3] = 0


# In[ ]:





# In[2]:


# for it in engd:
#     while len(it) < en1:
#         it.append('PAD')

# for it in frd:
#     while len(it) < fr1:
#         it.append('pad')


# In[3]:


# from keras.layers import RepeatVector
# model = Sequential()
# model.add(Embedding(input_dim = len(voc),output_dim = 20,input_length=en1,mask_zero=True))
# model.add(LSTM(256))
# model.add(RepeatVector(fr1))
# model.add(LSTM(256,return_sequences = True))
# model.add(Dense(len(voc1),activation='softmax'))
# model.compile('adam','categorical_crossentropy')


# In[4]:


# context = []
# target = frd
# for it in engd:
#     temp = []
#     for it1 in it:
#         temp.append(gg1[it1])
#     context.append(temp)


# In[5]:



        
# batch_size = 50
# d = 0
# while d<len(context):
#     print(d/batch_size)
#     curx = context[d:min(len(context),d+batch_size)]
#     cury = target[d:min(len(target),d+batch_size)]
#     cury2 = []
#     for tot in cury:
#         curyt = []
#         for it in tot:
#             cury1 = np.zeros(len(voc1))
#             cury1[gg3[it]] = 1
#             curyt.append(cury1)
#         cury2.append(curyt)
#     curx = np.array(curx)
#     cury2 = np.array(cury2)
#     print(curx.shape)
#     print(cury2.shape)
#     model.fit(curx,cury2,epochs = 250)
#     d += batch_size


# In[6]:


import json
f1 = open(path + '/Translator_English_w2i')
gg1 = json.load(f1)
gg = dict()
max1 = 0
for it in gg1:
    gg[gg1[it]] = it
#     max1 = max(max1,gg1[it])

# print(gg)
print(len(gg1))
print(len(gg))

f2 = open(path + '/Translator_French_w2i')
gg3 = json.load(f2)
gg2 = dict()
for it in gg3:
    gg2[gg3[it]] = it
print(max1)


# In[7]:


# model = keras.models.model_from_json(path + "Translator_4.json")
# model.load_weights(path + "Translator_4_weights.hdf5")
model = keras.models.load_model(path+"/Translator_4")


# In[8]:


import nltk
import random
# j3 = 0
# trans1 = []
# f3 = open("2019101096_MT1_train.txt","a")
# for it3 in engd:
#     trans = []
#     d = 0
#     temp = []
#     print(j3)
#     if len(it3) == 0:
#         f3.write("translated_sentence_" + str(j3+1) + "  " + '0' + '\n')
#         continue
#     while d < len(it3):
#         x1 = it3[d:min(len(it3),d + 24)]
#         while len(x1)<24:
#             x1.append("<PAD>")
#         x2 = []
#         for it in x1:
#             if it not in gg1:
#                 k1 = random.randint(1,19008)
#                 print(k1)
#                 x2.append(k1)
#             else:
#                 x2.append(gg1[it])
# #         for it in x2:
# #             if it > (len(gg1)-1):
# #                 it = 0
#         temp.append(x2)
# #         print(temp)
#         d += 24
# #     print(len(temp))
# #     print(temp)
#     matrix = model.predict(temp)
#     for it in matrix:
#         for it1 in it:
#             max1 = 0
#             maxi = 0
#             j = 0
#             for it2 in it1:
#                 if it2 > max1:
#                     max1 = it2
#                     maxi = j
#                 j += 1
#             if maxi == 0:
#                 break
#             trans.append(gg2[maxi])
# #     print(len(trans))
# #     print(len(frd[j3]))
# #     print(trans)
# #     print(frd[j3])
#     trans1.append(trans)
#     print("translated_sentence_" + str(j3+1) + "  " + str(nltk.translate.bleu_score.sentence_bleu(frd[j3],trans)) + '\n')
#     j3 += 1
# f3.write(nltk.translate.bleu_score.corpus_bleu(frd,trans1))

# cout<<tumhari maa de bj<<endl


# In[9]:


# print(str(nltk.translate.bleu_score.corpus_bleu(frd,trans1)))


# In[16]:


# f3 = open("./intro-to-nlp-assign3/ted-talks-corpus/test.en","r+")
# f4 = open("./intro-to-nlp-assign3/ted-talks-corpus/test.fr","r+")
# dat2 = f3.readlines()
# dat3 = f4.readlines()
# data2 = []
# data3 = []
# for it1 in dat2:
#     it = it1
#     it = re.sub("[\.]+"," ",it)
#     it = re.sub("[>]+"," ",it)
#     it = re.sub("[<]+"," ",it)
#     it = re.sub("[\,]+"," ",it)
#     it = re.sub("[\']+"," ",it)
#     it = re.sub("[\"]+"," ",it)
#     it = re.sub("[\;]+"," ",it)
#     it = re.sub("[\:]+"," ",it)
#     it = re.sub("[!]+"," ",it)
#     it = re.sub("[\?]+"," ",it)
#     it = re.sub("[\/]+"," ",it)
#     # re.sub("[\s]+","\s",it)
#     # f.write(it)
#     it = it.split()
#     data2.append(it)
# #     en1 = max(en1,len(it))

# for it1 in dat3:
#     it = it1
#     it = re.sub("[\.]+"," ",it)
#     it = re.sub("[>]+"," ",it)
#     it = re.sub("[<]+"," ",it)
#     it = re.sub("[\,]+"," ",it)
#     it = re.sub("[\']+"," ",it)
#     it = re.sub("[\"]+"," ",it)
#     it = re.sub("[\;]+"," ",it)
#     it = re.sub("[\:]+"," ",it)
#     it = re.sub("[!]+"," ",it)
#     it = re.sub("[\?]+"," ",it)
#     it = re.sub("[\/]+"," ",it)
#     # re.sub("[\s]+","\s",it)
#     # f.write(it)
#     it = it.split()
#     data3.append(it)
# #     fr1 = max(fr1,len(it))


# # In[18]:


# f5 = open("2019101096_MT1_test.txt","a")
# trans2 = []
# print(len(data2))
# print(len(data3))
# j3 = 0
# for it3 in data2:
#     trans = []
#     d = 0
#     temp = []
#     print(j3)
#     if len(it3) == 0:
#         f5.write("translated_sentence_" + str(j3+1) + "  " + '0' + '\n')
#         continue
#     while d < len(it3):
#         x1 = it3[d:min(len(it3),d + 24)]
#         while len(x1)<24:
#             x1.append("<PAD>")
#         x2 = []
#         for it in x1:
#             if it not in gg1:
#                 k1 = random.randint(1,19008)
#                 print(k1)
#                 x2.append(k1)
#             else:
#                 x2.append(gg1[it])
# #         for it in x2:
# #             if it > (len(gg1)-1):
# #                 it = 0
#         temp.append(x2)
# #         print(temp)
#         d += 24
# #     print(len(temp))
# #     print(temp)
#     matrix = model.predict(temp)
#     for it in matrix:
#         for it1 in it:
#             max1 = 0
#             maxi = 0
#             j = 0
#             for it2 in it1:
#                 if it2 > max1:
#                     max1 = it2
#                     maxi = j
#                 j += 1
#             if maxi == 0:
#                 break
#             trans.append(gg2[maxi])
# #     print(len(trans))
# #     print(len(frd[j3]))
# #     print(trans)
# #     print(frd[j3])
#     trans2.append(trans)
#     f5.write("translated_sentence_" + str(j3+1) + "  " + str(nltk.translate.bleu_score.sentence_bleu(data3[j3],trans)) + '\n')
#     j3 += 1
# print(str(nltk.translate.bleu_score.corpus_bleu(data3,trans2)))


# In[ ]:
sentence = input("Enter sentence:")
sentence = sentence.split()
trans = []
d = 0
temp = []
while d < len(sentence):
    x1 = sentence[d:min(len(sentence),d + 24)]
    while len(x1)<24:
        x1.append("<PAD>")
    x2 = []
    for it in x1:
        if it not in gg1:
            k1 = random.randint(1,19008)
            print(k1)
            x2.append(k1)
        else:
            x2.append(gg1[it])
#         for it in x2:
#             if it > (len(gg1)-1):
#                 it = 0
    temp.append(x2)
#         print(temp)
    d += 24
#     print(len(temp))
#     print(temp)
matrix = model.predict(temp)
for it in matrix:
    for it1 in it:
        max1 = 0
        maxi = 0
        j = 0
        for it2 in it1:
            if it2 > max1:
                max1 = it2
                maxi = j
            j += 1
        if maxi == 0:
            break
        trans.append(gg2[maxi])
str2 = ""
# print(trans)
for it in trans:
    str2 = str2 + (it) + " "


print(str2)



