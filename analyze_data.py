# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:57:12 2017

@author: papen
"""

import numpy as np
import quantities as pq
import elephant
import load_data as ld




def covariance_analysis(sts, 
                        binsize  = 150*pq.ms, 
                        binrange = [-0.3,0.3],
                        nbins    = 100):
    '''
    Performs a covariance analysis.
    
    INPUT:
        sts: spiketrains that have been annotated with 'exc', 'inh,
             or 'mix' indicating neuron type
        binsize: size of bins used for the analysis
        
    OUTPUT:
        pdf: dictionary of probability density distributions for 
             'exc', 'inh, or 'mix'
    '''
    covm = cross_covariance(sts, binsize=binsize)
    neu_types = get_neuron_types(sts)
    C   = dict()
    pdf = dict()
    
    for nty in set(neu_types):
        ids = np.where([neu_types[i]==nty for i in xrange(len(sts))])[0]
        pdf[nty], bins, C[nty] = get_pdf(covm, ids, binrange=binrange, nbins=nbins)
        
    return pdf, bins, C
        
        
    
    
    
def cross_covariance(sts, binsize, minNspk=3):
    '''
    Calculates cross-covariances between spike trains. 
    Auto-covariances are set to NaN.
    
    sts - array/list of N neo SpikeTrains
    binsize - quantity value (time), length of bin for spike train binning
    minNspk - minimal number of spikes in a spike train
    
    Returns a square array N x N of cross-covariances with the constrain, 
        that each spike train in correlated pair has to consist of at least 
        minNspk spikes, otherwise assigned value is NaN. 
    '''
    
    binned = elephant.conversion.BinnedSpikeTrain(sts, binsize = binsize)
    covm = elephant.spike_train_correlation.covariance(binned)
    
    for i, st in enumerate(sts):
        if len(st) < minNspk:
            covm[i,:] = np.NaN
            covm[:,i] = np.NaN
        covm[i,i] = np.NaN
    
    return covm
    


        
def get_pdf(C, ids, 
            binrange=[-0.3, 0.3], 
            auto_cross='cross', 
            nbins=100):
    Nunits, _ = np.shape(C)
    if auto_cross=='cross':
        tmp = np.copy(C)
        np.fill_diagonal(tmp, np.nan)
    if auto_cross=='auto':
        di = np.diag_indices(Nunits)
        tmp     = np.nan
        tmp[di] = C[di]
    Cout    = tmp[ids,:][:,ids]
    H, bins = np.histogram(Cout.ravel(), bins=nbins, 
                           range=binrange, density=True)
    bins = bins[1:]-(bins[1]-bins[0])/2
                           
    return H, bins, Cout
    
    
    
def get_neuron_types(sts):
    '''
    Checks neuron types of sts
    '''
    neu_types = []
    for i in xrange(len(sts)):
        neu_types.append(sts[i].annotations['neu_type'])
    return neu_types