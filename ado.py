# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:57:14 2020

@author: sun.fa
"""


import numpy as np


# =========================================
# sparsity-promoting optimiazations 
# =========================================
def FTRidge(X0, y, lam, maxit, tol, Mreg, normalize = 2, print_results = True):
    """
    This function is a single iteration of FTRidge
    
    """
    n, d = X0.shape
    X = np.zeros((n, d))
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros(d)
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i],normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else: X = X0
    
    # Get the standard ridge esitmate
    w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y))[0]

    num_relevant = d
    biginds = np.where(abs(np.multiply(Mreg, w)) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where(abs(np.multiply(Mreg, w)) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                if print_results: print("Tolerance too high - all coefficients set below tolerance")
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: 
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + 
                                         lam * np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
        else: 
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    
    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    if normalize != 0:
        return np.multiply(Mreg, w)
    else: return w


def TrainFTRidge(R0, Ut, tol, lam, eta, maxit = 200, FTR_iters = 10, l0_penalty = None, normalize = 0, split = 0.8, 
                 print_best_tol = False, plot_loss = False):
    """
    This function trains a predictor using FTRidge.
    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.
    
    R is Phi (evaluation of basis functions)
    Ut is f (simulation of the derivatives)
    
    """
    n,d = R0.shape
    R = np.zeros((n,d), dtype=np.float32)
    if normalize != 0:
        Mreg = np.zeros(d)
        for i in range(0,d):
            Mreg[i] = 1.0 / (np.linalg.norm(R0[:,i],normalize))
            R[:,i] = Mreg[i] * R0[:,i]                
        normalize_inner = 0
    else: 
        R = R0
        Mreg = np.ones(d)
        normalize_inner = 2
    

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train]
    TestY = Ut[test]

    # Set up l0 penalty
    if l0_penalty == None: l0_penalty = eta * np.linalg.cond(R)

    # Get the standard least squares estimator

    w_best = np.linalg.lstsq(TrainR, TrainY)[0]
    err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
    err_lambda = l0_penalty * np.count_nonzero(w_best)
    err_best = err_f + err_lambda
    
    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = FTRidge(TrainR, TrainY, lam, FTR_iters, tol, Mreg, normalize = normalize_inner)
        err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
        err_lambda = l0_penalty * np.count_nonzero(w)
        err = err_f + err_lambda
        
        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w

    return np.multiply(Mreg, w_best)

def STRidge(X0, y, lam, maxit, tol, Mreg, normalize = 2, print_results = True):
    """
    This function is a single iteration of STRidge
    
    """
    n, d = X0.shape
    X = np.zeros((n, d))
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros(d)
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i],normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else: X = X0
    
    # Get the standard ridge esitmate
    w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y))[0]

    num_relevant = d
    biginds = np.where(abs(np.multiply(Mreg, w)) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where(abs(np.multiply(Mreg, w)) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                if print_results: print("Tolerance too high - all coefficients set below tolerance")
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: 
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + 
                                         lam * np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
        else: 
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    
    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    if normalize != 0:
        return np.multiply(Mreg, w)
    else: return w


def TrainSTRidge(R0, Ut, lam, eta, d_tol, maxit = 200, STR_iters = 10, l0_penalty = None, normalize = 0, split = 0.8, 
                 print_best_tol = False, plot_loss = False):
    """
    This function trains a predictor using STRidge.
    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.
    
    R is Phi (evaluation of basis functions)
    Ut is f (simulation of the derivatives)
    
    """
    
    n,d = R0.shape
    R = np.zeros((n,d), dtype=np.float32)
    if normalize != 0:
        Mreg = np.zeros(d)
        for i in range(0,d):
            Mreg[i] = 1.0 / (np.linalg.norm(R0[:,i],normalize))
            R[:,i] = Mreg[i] * R0[:,i]                
        normalize_inner = 0
    else: 
        R = R0
        Mreg = np.ones(d)
        normalize_inner = 2
    

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train]
    TestY = Ut[test]
    
    #initialize threshold 
    d_tol = float(d_tol)
    tol = d_tol
    
    # Set up l0 penalty
    if l0_penalty == None: l0_penalty = eta * np.linalg.cond(R)

    # Get the standard least squares estimator

    w_best = np.linalg.lstsq(TrainR, TrainY)[0]
    err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
    err_lambda = l0_penalty * np.count_nonzero(w_best)
    err_best = err_f + err_lambda
    tol_best = 0
    
    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(TrainR, TrainY, lam, STR_iters, tol, Mreg, normalize = normalize_inner)
        err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
        err_lambda = l0_penalty * np.count_nonzero(w)
        err = err_f + err_lambda
        
        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol
        else:
            tol = max([0,tol - 2*d_tol])
            # d_tol  = 2*d_tol / (maxit - iter)
            d_tol  = d_tol / 1.618
            tol = tol + d_tol

    return np.multiply(Mreg, w_best), tol_best