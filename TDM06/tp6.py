# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:39:49 2015

@author: zwang04
"""

import yahmhelper as yhh
import yahmm as yh
import numpy as np

def EX1():
    data = np.load('TD4b-X1.npz') 
    sequences = data['sequences']
    nStates = int(data['nStates'])
    nObs = int(data['nObs'])
    
    s =0.25
    
    p0 =0.5+s*np.random.rand(nStates)
    p0/=p0.sum()
    
    A =0.5+s*np.random.rand ( nStates , nStates )
    A/=A.sum(axis =1,keepdims=True)
    
    B=0.5+s*np.random.randn(nStates,nObs)
    B/=B.sum(axis=1,keepdims =True)
    
    trainmodel=yhh.modelFromLambda(p0,A,B)
    return trainmodel

def EX2():
    data = np.load('TD4b-X2.npz') 
    x=data['x']
    y=data['y']
    #data = arr_0
    model = yh.Model()
    state0=yh.State(yh.NormalDistribution(-1,1))
    state1=yh.State(yh.NormalDistribution(1,2))
    
    model.add_state(state0)
    model.add_state(state1)
    
    model.add_transition(model.start,state0,1)
    model.add_transition(model.start,state1,0)
    model.add_transition(model.start,state1,0)
    
    model.add_transition(state1,state1,24/25)
    model.add_transition(state0,state1,1/50)
    model.add_transition(state1,state0,1/25)
    model.add_transition(state0,state0,49/50)
    model.bake()
    
    xEst=model.viterbi(y)
    (p0,A,B) = yhh.modelToLambda(model)
    return xEst
  
    
def main():
    model = EX1()
    xEst = EX2()
    
if __name__=="__main__":

    main()  