# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:34:28 2017

@author: papen
"""

from scipy.stats import levene
import sciunit
import numpy as np
import analyze_data as ana
import load_data
from quantities import ms

# ======================TEST SCORE AND TEST===============================

class LeveneScore(sciunit.Score):
    """
    A Levene Test score.
    Null Ho: homogeneity of variance or homoscedasticity
    
    """    
    @classmethod
    def compute(self, model):
        pvalue = levene(model.covar, exp.covar).pvalue
        self.score = LeveneScore(pvalue)
        return self.score
    
    _description = ("Levene's test is an inferential statistic used to assess the equality of variances. "
                  + "It tests the null hypothesis that the population variances are equal "
                  + "(called homogeneity of variance or homoscedasticity). "
                  + "If the resulting p-value of Levene's test is less than some significance level "
                  + "(typically 0.05), the obtained differences in sample variances are unlikely to have "
                  + "occurred based on random sampling from a population with equal variances.")
    
    @property
    def sort_key(self):
        return self.score
    
    def __str__(self):
        return 'pvalue = {:.3}'.format(self.score)
    

class CovarianceTest(sciunit.Test, LeveneScore):
    """
    Tests if the model has the same probability distribution of cross-
    covariances as observed in experimental data.
    """   
    required_capabilities = (ProducesSpikeTrains,)
    score_type = LeveneScore # This test's 'judge' method will return a Levenecore.
    
    def generate_prediction(self, model, verbose=False):
        """
        Implementation of sciunit.Test.general_prediction.
        Generates a prediction from a model using the required capabilities.
        """
        model.produce_covar()
        return model 
    
    def validate_observation(self, observation, first_try=True):
        """
        Implementation of sciunit.Test.validate_observation.
        Verifies that the pdfs are normalized to unit sum and have the same 
        number of bins (100). If not, it tries to transform them.
        """

        if not len(observation)==100:
            raise ValueError
        if np.round(np.sum(observation), 5) == 1.00000:
            pass
        else:
            observation = observation/float(np.sum(observation))
            pass
        

    def compute_score(self, model, p_value=.05, verbose=False):
        '''
        Implementation of sciunit.Test.compute_score.
        Compares the observation the test was instantiated with against the
        prediction returned in the previous step.
        This comparison of quantities is cast into a score.
        '''
        
        # get covars from experiment
        # set path
        datadir    = './'
        class_file = './simrest_validation/nikos2rs_consistency_EIw035complexc04.txt'
        
        # set parameters
        binsize  = 150*pq.ms
        nbins    = 100
        binrange = [-0.6, 0.6]
        eiThres  = 0.4
        
        # load data
        sts_mod = get_spiketrains(model)
        sts_exp = load_data.load_nikos2rs(path2file = datadir,
                                         eiThres    = eiThres,
                                         class_file = class_file)
        
        # calculate pdf of covariances
        pdf, bins, Cexp = ana.covariance_analysis(sts_exp, 
                                                  binsize  = binsize, 
                                                  binrange = binrange,
                                                  nbins    = nbins)
        pdf, bins, Cmod = ana.covariance_analysis(sts_mod, 
                                                  binsize  = binsize, 
                                                  binrange = binrange,
                                                  nbins    = nbins)
        l = LeveneScore.compute(Cmod.ravel(), Cexp.ravel()) 
        score = LeveneScore(l.score)
        score.description = "There is {} significant difference between the variances of inter spike intervals (p-value={})." \
                            .format('no' if score.score >= p_value else 'a', p_value)
        return score