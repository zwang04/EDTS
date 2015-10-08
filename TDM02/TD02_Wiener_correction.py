#!/usr/bin/env python3
# -*- coding: utf8 -*-
#
# Copyright 2014 Romain HERAULT <romain.herault@insa-rouen.fr>
#

import sys
pythonver=sys.version_info[0]


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as npl

import scipy,scipy.signal
import scipy.linalg as scl

import cvxpy as cp

if pythonver==2:
    import cPickle as pickle
else:
    import pickle


inspection=True
if inspection:
    #pour le debogage dans Spyder
    import inspect
    X1Avars = {}
    X1Bvars = {}
    X2vars  = []
    X3vars  = {}
#

def correlate(X,Y):
    """ Calcule la corrélation, applique la correction et retourne la partie paire """
    m=X.size
    n=Y.size
    
    # Caclul pondération
    correction=np.correlate(np.ones((m,)),np.ones((n,)),'full')
    # Corrélation complète
    rXY=np.correlate(X,Y,'full')
    # On pondère
    rXYc=rXY/correction
    # On garde la partie paire
    rXYc=rXYc[m-1:]
    
    return rXYc

def X1A(xtrain,ytrain,ytest,degre=10):
    """ Excercie 1 partie A"""
    
    # RYY h = rXY     
    rYY=correlate(ytrain,ytrain)
    rXY=correlate(xtrain,ytrain)
    
    # Ah=b
    A=scl.toeplitz(rYY[:degre])
    b=rXY[:degre]
    h=npl.inv(A).dot(b)
    
    # Estimation de X par filtrage h de Y
    xest=scipy.signal.lfilter(h,[1],ytest)
    
    plt.figure(1)
    plt.title('X1 A')
    plt.subplot(3,1,1)
    plt.ylabel('y')
    plt.plot(ytest,'b-')        
    plt.subplot(3,1,2)
    plt.ylabel('x estime')
    plt.plot(xest,'r-')
    plt.subplot(3,1,3)
    plt.ylabel('y, x estime')
    plt.plot(ytest,'b-')
    plt.plot(xest,'r-')    
    plt.savefig('X1A.pdf')
    plt.close()
    
    if inspection:
        global X1Avars
        X1Avars=inspect.currentframe().f_locals

def X1B(xtrain,btrain,ytest,degre=10):
    """ Excercie 1 partie B"""
    
    # (RXX+RBB) h = rXX         
    rXX=correlate(xtrain,xtrain)
    rBB=correlate(btrain,btrain)
    
    # Ah=b
    A=scl.toeplitz(rXX[:degre])+scl.toeplitz(rBB[:degre])
    b=rXX[:degre]
    h=npl.inv(A).dot(b)
    
    # Estimation de X par filtrage h de Y
    xest=scipy.signal.lfilter(h,[1],ytest)
    
    plt.figure(1)
    plt.title('X1 B')
    plt.subplot(3,1,1)
    plt.ylabel('y')
    plt.plot(ytest,'b-')        
    plt.subplot(3,1,2)
    plt.ylabel('x estime')
    plt.plot(xest,'r-')
    plt.subplot(3,1,3)
    plt.ylabel('y, x estime')
    plt.plot(ytest,'b-')
    plt.plot(xest,'r-')    
    plt.savefig('X1B.pdf')
    plt.close()
    
    if inspection:
        global X1Bvars
        X1Bvars=inspect.currentframe().f_locals         

def filtrePredicteur(y,n,degre):
    """ Filtre prédicteur pour un pas de n"""
    
    # RYY h = rYY[n:n+degre-1]            
    rYY=correlate(y,y)
    
    # Ah=b
    A=scl.toeplitz(rYY[:degre])
    b=rYY[n:n+degre]
    h=npl.inv(A).dot(b)
    
    return h

def X2(ytrain,ytest,degre=10):
    """ Excercie 2"""
    
    # Prédicition à 1 pas de temps puis à 50 pas de temps
    for n,nom in [(1,'A'),(50,'B')]:
    
        h=filtrePredicteur(ytrain,n,degre)
        
        # Estimation de Y[+n] par filtrage h de Y
        yest=scipy.signal.lfilter(h,[1],ytest)
        
        Ttest=np.arange(ytest.size)
        Test=np.arange(n,ytest.size+n)
        
        plt.figure(1)
        plt.title("Prediction a %d pas"%(n,))
        plt.plot(Ttest,ytest,'b-')
        plt.plot(Test,yest,'g-')
        plt.legend(('Signal','Prediction',),loc="upper right")
        plt.savefig('X2'+nom+'.pdf')
        plt.close()
        
        if inspection:
            global X2vars
            X2vars.append(inspect.currentframe().f_locals)


def quantileReg(t,y,alpha):
 
    # Données de la regression
    m=y.size
    n=2
    
 
    A=np.zeros((m,n))
    b=np.zeros((m,1))
    
    A[:,0]=t
    A[:,1]=np.ones((m,))
    
    b[:,0]=y

    # Variables
    x = cp.Variable(n)
    ep = cp.Variable(m)
    em = cp.Variable(m)
    
    # Constitution du problème
    objective = cp.Minimize(cp.sum_entries(cp.square(alpha*ep +(1.-alpha)*em)))
    constraints = [A*x-b+em-ep==0,ep>=0,em>=0]
    prob = cp.Problem(objective, constraints)
    
    # Résolution
    prob.solve()    
    xres=np.array(x.value)  
    
    return xres

            
def X3(yobs):
    
    # Calcul du temps correspondants aux observations
    nobs=yobs.size
    Tobs=np.arange(nobs)
    
    # Calcul des praramètres des droites encadrantes
    parambase=quantileReg(Tobs,yobs,0.9)
    paramcrete=quantileReg(Tobs,yobs,0.1)
    
    # Calcul des droites encadrantes pour les observations
    obsbase=Tobs*parambase[0]+parambase[1];
    obscrete=Tobs*paramcrete[0]+paramcrete[1];
    
    # Recadrage (correction) des observations
    yobscorr=(yobs-obsbase)/(obscrete-obsbase)
    yobsmean=yobscorr.mean()
    yobscorr-=yobsmean
    
    # On veut prédire sur 12 mois
    maxn=12
    ypredcorr=np.zeros((maxn,))
    Tpred=np.arange(nobs,nobs+maxn)
    
    # Pas besoin de prendre un degre élevé
    degre=4;
    #on doit faire un filtre qui prédit à +1 puis à +2 puis à +3 ...
    for i,n in enumerate(np.arange(1,maxn+1)):
        #calcule des coefficients pour predicteur +n
        h=filtrePredicteur(yobscorr,n,degre)
        #Application du filtre sur les degre dernières observations
        #Attention, il faut inverser l'ordre des coefficients avant de faire
        #le produit scalaire
        ypredcorr[i]=np.inner(h[::-1],yobscorr[-degre:])
 
    # Calcul des droites encadrantes pour la prédiction
    predbase=Tpred*parambase[0]+parambase[1];
    predcrete=Tpred*paramcrete[0]+paramcrete[1];
    # On inverse la correction    
    ypred=ypredcorr*(predcrete-predbase)+predbase+yobsmean

    # Signaux avec correction appliquée
    plt.figure(1)
    plt.title("Prediction annee suivante")
    plt.plot(Tobs,yobscorr,'b-')
    plt.plot(Tpred,ypredcorr,'g-')
    plt.legend(('Observation','Prediction',),loc="lower right")
    plt.savefig('X3corr.pdf')
    plt.close()
    
    # Signaux sans correction
    plt.figure(1)
    plt.title("Prediction annee suivante")
    plt.plot(Tobs,yobs,'b-')
    plt.plot(Tpred,ypred,'g-')
    plt.plot(Tobs,obsbase,'b:')
    plt.plot(Tobs,obscrete,'b:')
    plt.plot(Tpred,predbase,'g:')
    plt.plot(Tpred,predcrete,'g:')
    plt.legend(('Observation','Prediction',),loc="lower right")
    plt.savefig('X3.pdf')
    plt.close()


    if inspection:
        global X3vars
        X3vars=inspect.currentframe().f_locals

def main():
        data=pickle.load(open('TD02_Wiener_p%d.pkl'%(pythonver,),'rb'))
        
        #X1
        X1A(data['EX1']['train']['x'],data['EX1']['train']['y'],data['EX1']['test']['y'])
        X1B(data['EX1']['train']['x'],data['EX1']['train']['b'],data['EX1']['test']['y'])
        
        #X2
        X2(data['EX2']['train']['y'],data['EX2']['test']['y'])

        #X3
        X3(data['EX3']['test']['y'])

if __name__=="__main__":
    main()