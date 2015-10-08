# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:51:26 2015

@author: zwang04
"""

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import csv
import numpy as np

def loadFile(filename):
    fi=open(filename,'r')
    reader=csv.reader(fi,delimiter=' ')
    data=[]
    for row in reader:
        data.append([f for f in map(float,row)])
        return np.array(data)
        
obs = csv.reader(open('liste_bigrammes.csv','r',encoding='latin1'),delimiter='\t')
obs2 = csv.reader(open('liste_bigrammes.csv','r',encoding='latin1'),delimiter='\t')

i=0
C=[]
temp=''

for bi,dummy,count in obs:
    temp=temp+bi
#''.join(sorted(temp)) 

etatList=list(set(temp))
etatList.append('')
nbEtat = len(etatList)



C = np.zeros((nbEtat,nbEtat))



for bi,dummy,count in obs2:
    cx = etatList.index(bi[0])
    if len(bi)>1:
        cy = etatList.index(bi[1])
    else:
        cy = etatList.index('')
        
    C[cx,cy] = float(count)
    #print(count)
C.shape
W = C.sum(axis=1)
priorX0 = W / W.sum()
transitionMatrix = np.zeros((nbEtat, nbEtat))   
indices = W.nonzero()
transitionMatrix[indices] = C[indices] / W[indices, np.newaxis] 

def throwid(distrib):
    return np.argmax(np.random.multinomial(1, distrib))


def throwidseq(priorX0, transitionMatrix, longueurMot):
    nStates = priorX0.shape[0]
    seq = [throwid(priorX0)]
    for i in range(longueurMot - 1):
        if seq[-1] == nStates - 1:  # is endstate
            # endstate so word finishes here
            break
        seq.append(throwid(transitionMatrix[seq[-1]]))
    return seq


def idseq2str(aStateList, seq):
    nStates = len(aStateList)
    res = ''
    for s in seq:
        if s == nStates - 1:  # is endstate
            # endstate so word finishes here
            break
        res += aStateList[s]
    return res

for i in range(5):
    seq = throwidseq(priorX0, transitionMatrix, longueurMot=8)
    print(idseq2str(etatList, seq))
   # print (bi)
   # print (count)
    
#W=C.sum(axis=1, keepdims=True)
#with open ("liste_bigrammes.txt", "r") as myfile:
#    data=myfile.read()