#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import yahmm as yh

def modelFromLambda(p0,A,B):
    """
    Constructs a HMM based on parameters.
    You have to bake the model once return
    :param p0: prior distribution
    :type p0: (nStates,) numpy.ndarray
    :param A: dynamic model or transition model
    :type A: (nStates,nStates) numpy.ndarray
    :param B: mesurement model or emission model
    :type B: (nStates,nObs) numpy.ndarray
    :rtype: yahmm.Model
    """
    
    nStates=A.shape[0]
    nObs=B.shape[1]    
    
    model=yh.Model()
    
    states=[]
    for i in range(nStates):
        distparam=dict(zip(range(nObs),B[i]))
        dist=yh.DiscreteDistribution(distparam)
        states.append(yh.State(dist,name='{}-state'.format(i)))
        model.add_transition(model.start,states[i],p0[i])
    
    for i1,s1 in enumerate(states):
        for i2,s2 in enumerate(states):
            if A[i1,i2] != 0.:
                model.add_transition(s1,s2,A[i1,i2])
    
    return model

def modelToLambda(model):
    """
    Extracts prior, transition and emission distribution from an yahmm.Model
    :param model: HMM
    :type model: yahmm.Model
    :rtype: ((nStates,) numpy.ndarray, (nStates,nStates) numpy.ndarray, (nStates,nObs) numpy.ndarray)
    """    
    
    states=filter(lambda s:s.name.endswith('-state'),model.states)
    states=sorted(states,key=lambda s:int(s.name.split('-')[0]))
    
    nStates=len(states)
    nObs=len(states[0].distribution.parameters[0])
    
    A=np.zeros((nStates,nStates))
    B=np.zeros((nStates,nObs))
    p0=np.zeros((nStates,))
    
    for s1,s2,d in model.graph.edges(data=True):
        if (s2 in states):
            i2=states.index(s2)
            if s1==model.start:
                p0[i2]=np.exp(d['weight'])
            if (s1 in states):
                i1=states.index(s1)
                A[i1,i2]=np.exp(d['weight'])
            
    for s in states:
        if (s in states):
            i=states.index(s)
            for o in range(nObs):
                B[i,o]=s.distribution.parameters[0][o]
    
    return (p0,A,B)