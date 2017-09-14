# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:57:12 2017

@author: papen
"""

import numpy as np
import quantities as pq
import elephant




def covariance_analysis(sts, 
                        binsize  = 150*pq.ms, 
                        binrange = [-0.3,0.3], 
                        eiThres  = 0.4,
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
    neuron_type_separation(sts, eiThres=eiThres)
    covm = cross_covariance(sts, binsize=binsize)
    neu_types = get_neuron_types(sts)
    pdf  = dict()
    
    for nty in set(neu_types):
        ids = np.where([neu_types[i]==nty for i in xrange(len(sts))])[0]
        pdf[nty], bins = get_pdf(covm, ids, binrange=binrange, nbins=nbins)
        
    return pdf, bins
    
    
    

def neuron_type_separation(sts,
                           fname='./nikos2rs_consistency_EIw035complexc04.txt', 
                           eiThres=0.4):
    '''
    This function loads the consistencies for each unit.
    The consistencies are the percentages of single waveforms with 
    trough-to-peak times (t2p) larger than 350ms.
    
    Single units with small/large t2p are narrow/broad spiking units 
    that are putative inhibitory/excitatory units.
    
    The input neo SpikeTrain objects will be anotated with neu_type 'exc',
    'inh', or 'mix' if too many inconsistent waveforms are present 
    
    INPUT:
    eiThres [0-1]: threshold for the consistency. A small value will 
                   result in highly consistent waveforms. However, a
                   large amount of units will then not be classified.
                   
    OUTPUT:
    eIds: list of unit ids that are putative excitatory units
    iIds: list of unit ids that are putative inhibitory units                   
    '''
    Nunits = len(sts)
    consistency = np.loadtxt(fname,
                             dtype = np.float16)      
    exc = np.where(consistency > 1 - eiThres)[0]                       
    inh = np.where(consistency < eiThres)[0]
    mix = np.where(np.logical_and(consistency > eiThres,
                                  consistency < 1 - eiThres))[0]
    
    for i in exc:
        sts[i].annotations['neu_type'] = 'exc'
    for i in inh:
        sts[i].annotations['neu_type'] = 'inh'
    for i in mix:
        sts[i].annotations['neu_type'] = 'mix'
       
       
    print '\n## Classification of waveforms resulted in:'
    print '{}/{} ({:0.1f}%) neurons classified as putative excitatory'.format(
        len(exc), Nunits, float(len(exc))/Nunits*100.)
    print '{}/{} ({:0.1f}%) neurons classified as putative inhibitory'.format(
        len(inh), Nunits, float(len(inh))/Nunits*100.)
    print '{}/{} ({:0.1f}%) neurons unclassified (mixed)\n'.format(
        len(mix), Nunits, float(len(mix))/Nunits*100.)
#    return exc, inh, mix
        
        
    
    
    
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
                           
    return H, bins
    
    
    
def get_neuron_types(sts):
    '''
    Checks neuron types of sts
    '''
    neu_types = []
    for i in xrange(len(sts)):
        neu_types.append(sts[i].annotations['neu_type'])
    return neu_types