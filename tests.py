# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:34:28 2017

@author: papen
"""

from scipy.stats import levene
import sciunit
from quantities import ms

# ======================TEST SCORE AND TEST===============================

class LeveneScore(sciunit.Score):
    """
    A Levene Test score.
    Null Ho: homogeneity of variance or homoscedasticity
    
    """    
    @classmethod
    def compute(self, model1, model2):
        pvalue = levene(model1.spike_intervals,model2.spike_intervals).pvalue
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
    

class SpikeIntervalVariabilityTest(sciunit.Test, LeveneScore):
    """
    Tests if the model in simulator A has the same spike interval homoscedasticity as that in simulator B.
    Takes classes with attributes spiketrains which is a list of n Neo.SpikeTrains objects.
    Produced spike intervals are stored in a single array of floats (magnitude of times in ms).
    """   
    required_capabilities = (ProducesSpikeIntervals,) # since ProducesSpikeIntervals has only one method, one capability required for a model to take this test.  
    score_type = LeveneScore # This test's 'judge' method will return a Levenecore.
    
    def generate_prediction(self, model, verbose=False):
        """
        Implementation of sciunit.Test.general_prediction.
        Generates a prediction from a model using the required capabilities.
        """
        model.produce_intervals()
        return model 
    
    def validate_observation(self, observation, first_try=True):
        """
        Implementation of sciunit.Test.validate_observation.
        Verifies that the spike times are given in ms 
        and else tries to transform them.
        """
        if observation.spiketrains[0].units == ms:
            pass

        elif first_try:
            rescaled_spiketrains = []
            for st in observation.spiketrains:
                rescaled_spiketrains += [st.rescale('ms')]
            observation.spiketrains = rescaled_spiketrains
            self.validate_observation(observation, first_try=False)
        else:
            raise ValueError
            
    def compute_density_function(self, modelA, modelB, bins=100, show=True):
        """
        Additional function to visualize the features which are to be validated in this test.
        """
        if not len(modelA.spike_intervals):
            modelA.produce_intervals()
        if not len(modelB.spike_intervals):
            modelB.produce_intervals()
        # use the provided observation and the generated prediction to compute a score.
        hist_modelA, edges = np.histogram(modelA.spike_intervals, bins=bins, density=True)
        hist_modelB, ____  = np.histogram(modelB.spike_intervals, bins=edges, density=True)
        if show:
            dx = np.diff(edges)[0]
            xvalues = edges[:-1] + dx/2.0
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.tight_layout()
            ax.plot(xvalues, hist_modelA, label=modelA.name)
            ax.plot(xvalues, hist_modelB, label=modelB.name)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Inter-Spike Interval')
            ax.set_ylabel('Density')
            ax.legend();
            return fig
        else:
            return hist_modelA, hist_modelB

    def compute_score(self, modelA, modelB, p_value=.05, verbose=False):
        '''
        Implementation of sciunit.Test.compute_score.
        Compares the observation the test was instantiated with against the
        prediction returned in the previous step.
        This comparison of quantities is cast into a score.
        '''
        l = LeveneScore.compute(modelA, modelB) 
        score = LeveneScore(l.score)
        score.description = "There is {} significant difference between the variances of inter spike intervals (p-value={})." \
                            .format('no' if score.score >= p_value else 'a', p_value)
        return score