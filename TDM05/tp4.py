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
i=0
C=[]
temp=''

for bi,dummy,count in obs:
    temp=temp+bi
   # print (bi)
   # print (count)
    
    i=i+1
    
''.join(sorted(temp)) 

etatList=list(set(temp))
etatList.append('')
nbEtat = len(etatList)

C = np.zeros((nbEtat,nbEtat))

for bi,dummy,count in obs:
    cx = etatList.index(bi[0])
    if len(bi)>1:
        cy = etatList.index(bi[1])
    else:
        cy = etatList.index('')
        
    C[cx,cy] = count
    print(count)
    
   # print (bi)
   # print (count)
    
#W=C.sum(axis=1, keepdims=True)
#with open ("liste_bigrammes.txt", "r") as myfile:
#    data=myfile.read()