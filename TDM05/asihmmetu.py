#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class HMMModel(object):
    """
    HMM Model
    """
    def __init__(self,prior,transitions,emissions):
        """
        HMM Model constructor
        :param prior: prior distribution
        :type prior: (nStates,) numpy.Array
        :param transistions: dynamic model or transition model
        :type transitions: (nStates,nStates) numpy.Array
        :param emissions: mesurement model or emission model
        :type emissions: (nStates,nStates) numpy.Array
        """
        self._A=transitions
        """Transition model, (nStates,nStates) numpy.Array"""
        self._B=emissions
        """Emission model, (nStates,nStates) numpy.Array"""
        self._p0=prior
        """Prior distribution, (nStates,) numpy.Array"""
        
        if (len(self._A.shape) != 2) or (self._A.shape[0]!=self._A.shape[1]):
            raise ValueError("Transitions array is not square")
        
        if (len(self._B.shape) != 2):
            raise ValueError("Emissions array is not 2D")

        if (self._B.shape[0]!=self._A.shape[0]):
            raise ValueError("The number of states of the emissions array differs from transitions array")

        if (len(self._p0.shape) != 1):
            raise ValueError("Prior array is not 1D")

        if (self._p0.shape[0]!=self._A.shape[0]):
            raise ValueError("The number of states of the prior array differs from transitions array")

        self._nStates=self._A.shape[0]
        """Number of states in the model"""
        self._nObs=self._B.shape[1]
        """Number of obserations in the observation range according to the model"""
    
    def getNStates(self):
        """
        Get the number of states according to the model
        :returns: number of states
        :rtype: int
        """
        return self._nStates

    def getNObservations(self):
        """
        Get the observation range according to the model
        :returns: number of possible observation values
        :rtype: int
        """
        return self._nObs
    
    def getPrior(self):
        """
        Get the prior distribution according to the model
        :returns: prior distribution
        :rtype: (nStates,) numpy.Array
        """        
        return self._p0

    def getTransistions(self):
        """
        Get the dynamic model (or transition model) part of the model
        :returns: dynamic model
        :rtype: (nStates,nStates) numpy.Array
        """             
        return self._A
    
    def getEmissions(self):
        """
        Get the dynamic model (or emission model) part of the model
        :returns: dynamic model
        :rtype: (nStates,nStates) numpy.Array
        """           
        return self._B

    def forwardStateWeights(self,pastStateWeights):
        """
        Forward one step in time the state weights
        :param pastStateWeights: old state weights
        :type pastStateWeights: (nStates,) numpy.Array
        :returns: new state weights
        :rtype: (nStates,) numpy.Array
        """
        if (len(pastStateWeights.shape) != 1):
            raise ValueError("Past state weights is not 1D")
        if (pastStateWeights.shape[0]!=self._nStates):
            raise ValueError("The number of states in the past state weights differs from model's one")
        
        return pastStateWeights.dot(self._A)

    def forwardStateWeightsFKS(self,knownState):
        """
        From a known state, forward one step in time the state weights
        :param knownState: old known state 
        :type knownState: int
        :returns: new state weights
        :rtype: (nStates,) numpy.Array
        """
        return self._A[knownState]


    def backwardStateWeights(self,currentStateWeights):
        """
        Forward one step in time the state weights
        :param currentStateWeights: current state weights
        :type currentStateWeights: (nStates,) numpy.Array
        :returns: projected old state weights
        :rtype: (nStates,) numpy.Array
        """
        if (len(currentStateWeights.shape) != 1):
            raise ValueError("Current state weights is not 1D")
        if (currentStateWeights.shape[0]!=self._nStates):
            raise ValueError("The number of states in the current state weights differs from model's one")
        
        return self._A.dot(currentStateWeights)

    def backwardStateWeightsFKS(self,knownState):
        """
        From a known state, backward one step in time the state weights
        :param knownState: current known state 
        :type knownState: int
        :returns: projected old state weights
        :rtype: (nStates,) numpy.Array
        """
        return self._A[:,knownState]

 
    def deduceObservationWeight(self,stateDist):
        """
        Deduce observation weights from the state weights
        :param stateDist: state weights
        :type stateDist: (nStates,) numpy.Array
        :returns: emission weights
        :rtype: (nObs,) numpy.Array
        """          
        if (len(stateDist) != 1):
            raise ValueError("State weights is not 1D")
        if (stateDist.shape[0]!=self._nStates):
            raise ValueError("The number of states in the state weights differs from model's one")
        
        return stateDist.dot(self._B) 

    def deduceObservationWeightFKS(self,knownState):
        """
        From a known state, deduce observation weights
        :param knownState: known state 
        :type knownState: int
        :returns: observation weights
        :rtype: (nObs,) numpy.Array
        """          
        return self._B[knownState]

    def induceStateWeightsFKO(self,knownObservation):
        """
        From a known observation, induce state weights
        :param knownObservation: known observation 
        :type knownObservation: int
        :returns: state weights
        :rtype: (nStates,) numpy.Array
        """          
        return self._B[:,knownObservation]
    
    def sampleSequence(self,n):
        """
        Generate a sequence of n samples from the model
        :returns: (state sequence, observation sequence )
        :rtype: ( (n,) numpy.Array, (n,) numpy.Array )
        """
        
        #Allocation
        #States sequence
        x=np.zeros((n+1,))
        #Observation sequence
        y=np.zeros((n,))
        
        #Initialization
        x[0]=np.random.multinomial(1,self._p0).argmax()
        #Recursion
        for i in range(n):
            #For states: i past state, i+1 current state
            #For observations: i current observation
            #Given the past state, throw a new state
            raise NotImplementedError
            #Given the current state, throw a new observation
            raise NotImplementedError

        return (x,y)
    

class HMMSequence(object):
    """
    HMM Sequence
    """    
    def __init__(self,model,y):
        """
        HMM Sequence constructor
        :param model: the HMM model
        :type model: HMMModel
        :param y: observation sequence, n observation samples, each sample must be in the observation range of the model.
        :type y: (n,) numpy.Array
        """
        self._model=model
        """The HMM model"""    
        self._y=y
        """Observation sequence, (_n,) numpy.Array"""
        if (len(self._y.shape) != 1):
            raise ValueError("Y is not a 1-D array")
        
        #Set of possible observations
        obsSet=set(range(self._model.getNObservations()))
        #Set of "observed" observations
        for o in set(y):
            if not (o in obsSet):
                raise ValueError("Sequence observation {} is not in observation range from the model".format(i))
            
        
       
        #Private attributes
        self._n=self._y.shape[0]
        """Sample number in the sequence"""
        self._alpha=None
        """(_n+1,model._nStates) numpy.array fills with p(\vy_{1:k},\vx_k=i|\lambda)"""
        self._likelihood=None
        """Sequence likelihod  p(\vy_{1:T}|\lambda)"""
        
        self._beta=None
        """(_n+1,model._nStates) numpy.array fills with p(\vy_{k+1:T}|\vx_k=i,\lambda)"""
        self._gammanormterm=None
        self._gamma=None
        """(_n+1,model._nStates) numpy.array fills with p(\vx_k=i|\lambda,\vy_{1:T})"""
        self._xi=None
        """(_n,model._nStates) numpy.array fills with p(\vx_k=i,\vx_{k+1}=j|\lambda,\vy_{1:T})"""

        self._delta=None
        """(_n+1,model._nStates) numpy.array fills with max_{\vx_{0:k-1}} p(\vy_{1:k},\vx_{0:k-1},\vx_k=i) """
        self._psi=None
        """(_n,model._nStates) numpy.array [k,j] fills with argmax_{i} \delta_{k-1}(i)  p(\vx_{k}=j|\vx_{k-1}=i)"""
        self._estx=None
        """(_n+1,) numpy.array fills with argmax_{\vx_{0:k-1}} p(\vy_{1:k},\vx_{0:k-1},\vx_k=i)"""

        
    def _forward(self):
        """
        Do the forward pass, fills self._alpha
        """        
        #Model infos
        A=self._model.getTransistions()
        B=self._model.getEmissions()
        #Allocation and recursion init
        self._alpha=np.zeros((self._n+1,self._model.getNStates()))
        self._alpha[0]=self._model.getPrior()
        #Recursion
        for k in range(self._n):
                raise NotImplementedError

                
    def _backward(self):
        """
        Do the backward pass, fills self._beta
        """              
        #Model infos
        A=self._model.getTransistions()
        B=self._model.getEmissions()
        #Allocation and recursion init
        self._beta=np.zeros((self._n+1,self._model.getNStates()))
        self._beta[-1]=np.ones((self._model.getNStates(),))
        self._beta[-2]=self._model.backwardStateWeights(self._beta[-1])
        #Recursion
        for k in reversed(range(self._n-1)):
                raise NotImplementedError
        
    def _track(self):
        """
        Do the track pass, fills self._delta and self._psi
        """            
        #Model infos
        A=self._model.getTransistions()
        B=self._model.getEmissions()
        #Allocation
        self._delta=np.zeros((self._n+1,self._model.getNStates()))
        self._psi=np.zeros((self._n,self._model.getNStates()))
        #Recursion init
        self._delta[0]=self._model.getPrior()
        #Recursion
        for k in range(self._n):
            raise NotImplementedError


    def _backtrack(self):
        """
        Do the backtrack pass, fills self._estx, the estimated state sequence
        """          
                
        #Sequence infos
        delta=self.getDelta()
        psi=self.getPsi()
        #Allocation and recursion init
        self._estx=np.zeros((self._n+1,))
        self._estx[-1]=delta[-1].argmax()
        #Recursion
        for k in reversed(range(self._n)):
            raise NotImplementedError
    
    def getAlpha(self):
        """
        Get alphas of the sequence 
        :returns: array [k,i] of  p(\vy_{1:k},\vx_k=i|\lambda)
        :rtype: (_n+1,model._nStates) numpy.array 
        """
        if self._alpha is None:
            self._forward()
        return self._alpha


    def getLikelihood(self):    
        if self._likelihood is None:
            alpha=self.getAlpha()
            #Compute _likelihood
            self._likelihood=alpha[-1].sum()
        return self._likelihood 

    def getBeta(self):
        """
        Get betas of the sequence 
        :returns: array [k,i] of  p(\vy_{k+1:T}|\vx_k=i,\lambda)
        :rtype: (_n+1,model._nStates) numpy.array 
        """        
        if self._beta is None:
            self._backward()
        return self._beta
        
    def getGamma(self):
        """
        Get gammas of the sequence 
        :returns: array [k,i] of  p(\vx_k=i|\lambda,\vy_{1:T})
        :rtype: (_n+1,model._nStates) numpy.array 
        """           
        if self._gamma is None:
            alpha=self.getAlpha()
            beta=self.getBeta()
            #Raw gammas
            gammanonorm=alpha*beta
            #Normalization terms (one per time step)
            self._gammanormterm=1./gammanonorm.sum(axis=1)
            #Normalized gammas
            self._gamma=np.diag(self._gammanormterm).dot(gammanonorm)
        return self._gamma

    def getXi(self):
        """
        Get xis of the sequence 
        :returns: array [k,i,j] of  p(\vx_k=i,\vx_{k+1}=j|\lambda,\vy_{1:T})
        :rtype: (_n,model._nStates) numpy.array 
        """           
        if self._xi is None:
            if self._gammanormterm is None:
                self.getGamma()
            alpha=self.getAlpha()
            beta=self.getBeta()
            A=self._model.getTransistions()
            B=self._model.getEmissions()            
            #Raw gammas
            self._xi=np.zeros((self._n,self._model.getNStates(),self._model.getNObservations()))
            for k in range(self._n):
                for i in range(self._model.getNStates()):
                    for j in range(self._model.getNObservations()):
                        self._xi[k,i,j]=alpha[k,i]*A[i,j]*B[j,self._y[k]]*beta[k+1,j]
        return self._xi

    def getDelta(self):
        """
        Get deltas of the sequence 
        :returns: array [k,i] of  max_{\vx_{0:k-1}} p(\vy_{1:k},\vx_{0:k-1},\vx_k=i)
        :rtype: (_n+1,model._nStates) numpy.array 
        """           
        if self._delta is None:
            self._track()
        return self._delta

    def getPsi(self):
        """
        Get psis of the sequence 
        :returns: array [k,j] of  argmax_{i} \delta_{k-1}(i)  p(\vx_{k}=j|\vx_{k-1}=i)
        :rtype: (_n+1,model._nStates) numpy.array 
        """
        if self._psi is None:
            self._track()
        return self._psi
    
    def getEstimatedX(self):
        """
        Get estimated states coresponding to observations 
        :returns:  estimated states argmax_{\vx_{0:k-1}} p(\vy_{1:k},\vx_{0:k-1},\vx_k=i)
        :rtype: (_n+1,) numpy.array 
        """        
        if self._estx is None:
            self._backtrack()
        return self._estx


    
