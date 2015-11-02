#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014 Romain HERAULT <romain.herault@insa-rouen.fr>
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
if sys.version_info < (3,):
    reload(sys)  # Reload does the trick!
    sys.setdefaultencoding('UTF8')

if sys.version_info < (3,):
    String = unicode
    StringType = 'unicode'
    Binary = str
    BinaryType = 'str'
else:
    String = str
    StringType = 'str'
    Binary = bytes
    BinaryType = 'bytes'

import numpy as np


def filter_predict(transition_function, transition_function_jacobian, transition_covariance,
                   current_state_mean, current_state_covariance):
    """
        Compute the predict step of a first order Extended Kalman Filter

        :param transition_function: python function that applies one step in the dynamical model
        :type transition_function: (nStateDim,) numpy.ndarray -> (nStateDim,) numpy.Array
        :param transition_function_jacobian: python function that computes the jacobian of the transition_function for a given state
        :type transition_function_jacobian: (nStateDim,) numpy.ndarray -> (nStateDim,nStateDim) numpy.Array
        :param transition_covariance: covariance of the additive noise in the dynamical model
        :type transition_covariance: (nStateDim,nStateDim) numpy.Array
        :param current_state_mean: current state mean
        :type current_state_mean: (nStateDim,) numpy.ndarray        
        :param current_state_covariance: current state covariance
        :type current_state_covariance: (nStateDim,nStateDim) numpy.Array
        :returns: predicted state mean, predicted state covariance
        :rtype: ( (nStateDim,) numpy.Array, (nStateDim,nStateDim) numpy.ndarray )
    """
    predicted_state_mean = transition_function(current_state_mean)

    transition_matrix = transition_function_jacobian(current_state_mean)
    predicted_state_covariance = transition_matrix.dot(
        current_state_covariance).dot(transition_matrix.T) + transition_covariance

    return (predicted_state_mean, predicted_state_covariance)


def filter_update(observation_function, observation_function_jacobian, observation_covariance,
                  predicted_state_mean, predicted_state_covariance, observation):
    """
        Compute the update step of a first order Extended Kalman Filter

        :param observation_function: python function that applies the observation model on a given state
        :type observation_function: (nStateDim,) numpy.ndarray -> (nObservationDim,) numpy.Array
        :param observation_function_jacobian: python function that computes the jacobian of the observation_function for a given state
        :type observation_function_jacobian: (nStateDim,) numpy.ndarray -> (nObservationDim,nStateDim) numpy.Array
        :param observation_covariance: covariance of the additive noise in the observation model
        :type observation_covariance: (nObservationDim,nObservationDim) numpy.Array
        :param predicted_state_mean: predicted state mean
        :type predicted_state_mean: (nStateDim,) numpy.ndarray        
        :param predicted_state_covariance: predicted state covariance
        :type predicted_state_covariance: (nStateDim,nStateDim) numpy.Array
        :param observation: current observation
        :type observation: (nObservationDim,) numpy.ndarray        
        :returns: new_state_mean, new state covariance
        :rtype: ( (nStateDim,) numpy.Array, (nStateDim,nStateDim) numpy.ndarray )
    """
    error = observation - observation_function(predicted_state_mean)

    observation_matrix = observation_function_jacobian(predicted_state_mean)

    S = observation_matrix.dot(predicted_state_covariance).dot(
        observation_matrix.T) + observation_covariance
    K = predicted_state_covariance.dot(
        observation_matrix.T).dot(np.linalg.inv(S))

    new_state_mean = predicted_state_mean + K.dot(error)
    new_state_covariance = predicted_state_covariance - K.dot(S).dot(K.T)

    return (new_state_mean, new_state_covariance)
