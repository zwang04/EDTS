#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014 Romain HERAULT <romain.herault@insa-rouen.fr>
#

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
from pykalman import KalmanFilter


def main():

    #Paramètres du modèle
    #Indiquer à quoi correspond chaque variable
    osigma=0.1;

    transition_matrix = np.array([[1., 0.,0.],[1., 1.,0.],[0.,0,0.9]])
    transition_covariance = np.zeros((3,3));

    observation_matrix = np.array([[0., 1.,0.],[0., 0.,1.]])
    observation_covariance = np.eye(2)*osigma

    initial_state_mean = np.array([1,0,10])
    initial_state_covariance = np.eye(3);
    
    
    #Observations
    observations=np.array([ [1.1,9.2],
                            [1.9,8.1],
                            [2.8,7.2],
                            [4.2,6.6],
                            [5.0,5.9],
                            [6.1,5.32],
                            [7.2,4.7],
                            [8.1,4.3],
                            [9.0,3.9]])
    

    # Filtrage à la main
    # Quels sont les paramètres du constructeur ?
    kf = KalmanFilter(
        transition_matrix, observation_matrix,
        transition_covariance, observation_covariance,
    )    
    
    # Que conserverons les variables suivantes ?
    hand_state_estimates=[initial_state_mean]
    hand_state_cov_estimates=[initial_state_covariance]
    
    for anObs in observations:
        # Quelles étapes du filtrage sont réalisées par la ligne suivante
        (aMean,aCov) = kf.filter_update(hand_state_estimates[-1],hand_state_cov_estimates[-1],anObs)
        hand_state_estimates.append(aMean)
        hand_state_cov_estimates.append(aCov)
    hand_state_estimates=np.array(hand_state_estimates)
    
    # A quoi sert la ligne suivante ?
    hand_positions=np.dot(hand_state_estimates,observation_matrix.T )    
    
    plt.figure(1)
    plt.plot(observations[:,0],observations[:,1], 'r+')
    plt.plot(hand_positions[:,0],hand_positions[:,1], 'b')
    plt.savefig('illustration_filtrage_main.pdf')
    plt.close()


    # Filtrage complet
    # Que fait cette section ?
    # Quels sont les paramètres supplémentaires donnés au constructeur ?
    # Quels sont les autres paramètres possibles ?
    kf = KalmanFilter(
        transition_matrix, observation_matrix,
        transition_covariance, observation_covariance,
        initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance,
    )

    
    (filtered_state_estimates,filtered_state_cov_estimates) = kf.filter(observations)
    filtered_positions=np.dot(filtered_state_estimates,observation_matrix.T )
    
    plt.figure(1)
    plt.plot(observations[:,0],observations[:,1], 'r+')
    plt.plot(filtered_positions[:,0],filtered_positions[:,1], 'b')
    plt.savefig('illustration_filtrage.pdf')
    plt.close()

    # Lissage
    # Que fait cette section ?
    # Quel est la différence avec le filtrage ?
    # Puis-je faire un lissage "à la main" observation par observation ?
   
    kf = KalmanFilter(
        transition_matrix, observation_matrix,
        transition_covariance, observation_covariance,
        initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance,
    )

    (smoothed_state_estimates,smoothed_state_cov_estimates) = kf.smooth(observations)
    smoothed_positions=np.dot(smoothed_state_estimates,observation_matrix.T )
    
    plt.figure(2)
    plt.plot(observations[:,0],observations[:,1], 'r+')
    plt.plot(smoothed_positions[:,0],smoothed_positions[:,1], 'b')
    plt.savefig('illustration_lissage.pdf')
    plt.close()


if __name__=="__main__":
    main()