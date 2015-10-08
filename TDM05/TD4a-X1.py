#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np

filepath = 'liste_bigrammes.csv'


def loadFile(filepath):
    # initialize two empty list
    bigrams, counts = [], []
    # we affecte fd by the "with" statement
    # so fd is automaticly destroy outside the "with"
    with open(filepath, 'r', encoding='latin1') as fd:
        reader = csv.reader(fd, delimiter='\t')
        for aBigram, dummy, aCount in reader:
            bigrams.append(aBigram)
            counts.append(float(aCount))
    return bigrams, counts


def computeModel(bigrams, counts):
    
    #Get all possible chars from the bigrams
    chars = sorted(list(set(''.join(bigrams))))
    #Put an "End State" at the end of the list
    #If we encouter a end state, then the word is finished
    aStateList = chars + ['ENDSTATE']
    nStates = len(aStateList)

    C = np.zeros((nStates, nStates))

    for aBigram, aCount in zip(bigrams, counts):
        currentId = aStateList.index(aBigram[0])
        if len(aBigram) > 1:
            nextId = aStateList.index(aBigram[1])
        else:
            #if the bigram is only 1 char, it means
            #that the word finishes here
            nextId = nStates - 1  # EndState Id
        C[currentId, nextId] = aCount

    W = C.sum(axis=1)
    #That is a fake estimation of the prior...
    priorX0 = W / W.sum()

    transitionMatrix = np.zeros((nStates, nStates))
    ## Numpy tricks
    # get indices of non-null elements of W
    indices = W.nonzero()
    #Divide C line by line where the corresponging W if W non-null
    # transitionMatrix[indices] shape is (x,nStates)
    # C[indices] shape is (x,nStates)
    # W[indices] shape is (x,);  W[indices, np.newaxis] shape is (x,1)
    # a (x,1) array is "broadcastable" to a (x,nStates) array
    # a (x,) array is not
    transitionMatrix[indices] = C[indices] / W[indices, np.newaxis] 


    return aStateList,  priorX0, transitionMatrix


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


def main():

    bigrams, counts = loadFile(filepath)
    aStateList, priorX0, transitionMatrix = computeModel(bigrams, counts)

    for i in range(5):
        seq = throwidseq(priorX0, transitionMatrix, longueurMot=8)
        print(idseq2str(aStateList, seq))


if __name__ == "__main__":
    main()
