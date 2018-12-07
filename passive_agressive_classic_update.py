# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:12:55 2018

Implementation of the passive-aggressive online learning algorithm for binary
classification. It can be seen as extension of SVM to the context of online
learning. The principle followed here is to update the solution in order to
stay as close as possible to the current one while achieving at least a unit
margin on the most recent example

@author: deric
"""

import numpy as np


def passive_agressive_online(X, y, impl, C=None):
    """Learn a weight vector from the data matrix X using a passive-agressive
    learning algorithm"""
    
    w = np.zeros(X.shape[1])
    for t in range(X.shape[0]):
        # Receive instance x_t
        x_t = X[t, :]
        # Prediction
        # y_t_pred = np.sign(np.dot(w, x_t))
        # Receive label y_t
        y_t = y[t]
        # Compute loss l_t
        l_t = max([0, 1 - y_t*np.dot(w, x_t)])
        # Compute r_t
        if impl == "classic":
            r_t = l_t / (np.linalg.norm(x_t)**2)
        elif impl == "relax1":
            r_t = min([C, l_t / (np.linalg.norm(x_t)**2)])
        elif impl == "relax2":
            r_t = l_t / ((np.linalg.norm(x_t)**2) + 1/(2*C))
        # Compute update
        w += r_t * y_t * x_t
        
    return w
