# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:00:55 2015

@author: zwang04
"""

#source /uv/asi/edts/venvp3/bin/activate

#import pickle

#data = pickle.local(open('~/Bureau/ASI5/EDTS/TDM02/','rb'))
import pickle
import numpy as np
from scipy.signal  import butter, lfilter, tf2ss
import scipy as sc
import matplotlib as ma
import scipy.linalg as scl
import matplotlib.pyplot as plt

main()

#1 Péliminiares
#1 intercorrélation x ,y

def correlate(X,Y):
    m=X.size
    n=Y.size
    correction=np.correlate(np.ones((m,)),np.ones((n,)),'full')    
    rXY=np.correlate(X,Y,'full')
    rXYc=rXY/correction
    rXYC=rXYc[m-1:]
    return rXYc

#2 FIR filter
#lfilter scipy.signal.lfilter(b, a, x, axis=-1, zi=None)

#a=1
#lfilter(b,a,axis=-1,zi=None)

# x,y
def exa1(data):
    y1=data['EX1']['train']['y']
    #b1=data['EX1']['train']['b']
    x1=data['EX1']['train']['x']
    Ry=correlate(y1,y1)
    rxy=correlate(x1,y1)
    h1=rxy/Ry # wrong
    return h1
    
# x,b
def exa2(data,h):
    y2=data['EX1']['test']['y']
    #b1=data['EX1']['train']['b']
    #x1=data['EX1']['train']['x']
    xEstime=lfilter(h,[1,],y2,axis=-1, zi=None)    

def X1A(xtrain,ytrain,ytest,degree):
    # RYY h =rXY
    rYY=correlate(ytrain,ytrain)
    rXY=correlate(xtrain,ytrain)
    
    # Ah = b
    A=sc.linalg.toeplitz(rYY[:degree])
    b=rXY[:degree]
    h=sc.linalg.inv(A).dot(b)
    
    #estimation de X par filtrage h de Y
    xest=lfilter(h,[1],ytest)
    
    plt.figure(1)
    plt.title('X1 A')
    plt.subplot(3,1,1)
    plt.ylabel('y')
    plt.plot(ytest,'b-')
    plt.subplot(3,1,2)
    plt.ylabel('x estimate')
    plt.plot(xest,'r-')
    plt.subplot(3,1,3)
    plt.ylabel('y, xestimate')
    plt.plot(xest,'b-')
    plt.savefig('ex1.png')
    
def X1B(xtrain,b):
    # RX + RB h =rX
    RX=correlate(xtrain,xtrain)
    RB=correlate(b,b)

    
    # Ah = b
    A=sc.linalg.toeplitz(RX[:degree])
    rX=RX[1:]
    #b=rXY[:degree]
    #h=sc.linalg.inv(A).dot(b)
    
    
def main():
    data=pickle.load(open('/home/zwang04/Bureau/ASI5/EDTS/TDM02/TD02_Wiener_p3.pkl','rb'))
    #h = exa1(data)
    #xEstime = exa2(data,h)
    
    y1=data['EX1']['train']['y']
    x1=data['EX1']['train']['x']
    y2=data['EX1']['test']['y']
    degree = 10
    X1A(x1,y1,y2,degree)
