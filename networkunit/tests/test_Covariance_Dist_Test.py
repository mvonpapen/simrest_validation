import sciunit
import sciunit.scores
import networkunit.capabilities as cap
import networkunit.plots as plots

import quantities
import os

import neo

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt




# COMMENTS ##############################
#
# all open issues marked with "## TODO ##"
#
# Capabilities: Produces_SpikeTrains
#
# COMMENTS ##############################


#==============================================================================

class Covariance_Dist_Test(sciunit.Test):
    """
    Tests for equal variances of cross-covariances of neural network simulation
    against experimental data obtained from Utah array in (pre)motor cortex of 
    macaque monkey during resting state
    """
    score_type = ## TODO ##
    id = ## TODO ##

    def __init__(self,
                 observation={},
                 name="Covariance dist. - resting state - motor cortex"):
        description = ("Tests the covariance distribution of motor cortex "
                       +"during resting state")
        self.units = quantities.um
        required_capabilities = (cap.Produces_SpikeTrains,)

        observation = self.format_data(observation)
        observation, pdf_obs, bins_obs = self.covariance_analysis(observation)
        self.figures = []
        sciunit.Test.__init__(self, observation, name)

        self.directory_output = './output/'

    #----------------------------------------------------------------------

    def format_data(self, data):
        """
        This accepts data input in the form:
        ***** (observation) *****
        list of Nunit neo.SpikeTrains with annotations 'exc', 'inh'
        ***** (prediction) *****
        list of Nunit neo.SpikeTrains with annotations 'exc', 'inh'
        """
        neu_types = []
        for st in data:
            try: # neo SpikeTrains?
                assert type(st) is neo.core.spiketrain.SpikeTrain
            except:
                raise sciunit.Error("List elements are not "
                                    +"neo.core.spiketrain.SpikeTrains.")
            try: # has neu_type annotations?
                assert 'neu_type' in st.annotations
            except:
                raise sciunit.Error("SpikeTrain has no neu_type annotation.")
            neu_types.append(st.annotations['neu_type'])
        neu_types = set(neu_types)
        try:
            assert any('exc' == s for s in neu_types)
        except:
            raise sciunit.Error("There are no exc units.")
        try:
            assert any('inh' == s for s in neu_types)
        except:
            raise sciunit.Error("There are no inh units.")
        return data

    #----------------------------------------------------------------------
    ## TODO ##
#    def validate_observation(self, observation):
#        try:
#            for key0 in observation.keys():
#                for key, val in observation["diameter"].items():
#                    assert type(observation["diameter"][key]) is quantities.Quantity
#        except Exception:
#            raise sciunit.ObservationError(
#                ("Observation must return a dictionary of the form:"
#                 "{'diameter': {'mean': 'XX um', 'std': 'YY um'}}"))

    #----------------------------------------------------------------------

    def generate_prediction(self, model, verbose=False):
        """Implementation of sciunit.Test.generate_prediction."""
        self.model_name = model.name
        prediction = model.get_sts() ## TODO ##
        prediction = self.format_data(prediction)
        prediction, pdf_pre, bins_pre = self.covariance_analysis
        return prediction

    #----------------------------------------------------------------------

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""
        print "observation = ", observation
        print "prediction = ", prediction
        self.score = sciunit.scores.ZScore.compute(observation["diameter"], prediction["diameter"])
        self.score.description = "A simple Z-score"

        # create output directory
        self.path_test_output = self.directory_output + 'soma_diameter_mean_sd/' + self.model_name + '/'
        if not os.path.exists(self.path_test_output):
            os.makedirs(self.path_test_output)

        self.observation = observation
        self.prediction = prediction
        # create relevant output files
        # 1. Error Plot
        err_plot = plots.ErrorPlot(self)
        err_plot.xlabels = ["Soma"]
        err_plot.ylabel = "Diameter (um)"
        file1 = err_plot.create()
        self.figures.append(file1)
        # 2. Text Table
        txt_table = plots.TxtTable(self)
        file2 = txt_table.create()
        self.figures.append(file2)
        return self.score

    #----------------------------------------------------------------------

    def bind_score(self, score, model, observation, prediction):
        score.related_data["figures"] = self.figures
        return score



    #---Functions needed to compute distribution of cov from spiketrains---
        
    def covariance_analysis(sts, 
                            binsize  = 150*pq.ms, 
                            binrange = [-0.4,0.4],
                            nbins    = 80):
        '''
        Performs a covariance analysis of annotated spiketrains.
        INPUT:
            sts: list of N spiketrains that have been annotated (exc/inh)
            binsize: quantities value for binned spiketrain
            binrange: binrange used for histogram
            nbins: number of bins within binrange
        OUTPUT:
            C: dictionary of exc/inh containing elements covariance matrices
            pdf: dictionary of probability density distributions for 
                 'exc' and 'inh'
            bins: bin centers of pdf
        '''
        covm = cross_covariance(sts, binsize=binsize)
        neu_types = get_neuron_types(sts)
        C   = dict()
        pdf = dict()        
        for nty in set(neu_types):
            ids = np.where([neu_types[i]==nty for i in xrange(len(sts))])[0]
            C[nty], pdf[nty], bins = get_ei_covar_pdf(covm, ids, 
                                                      binrange=binrange, 
                                                      nbins=nbins)       
        return C, pdf, bins
            
        
        
    def cross_covariance(sts, binsize, minNspk=3):
        '''
        Calculates cross-covariances between spike trains. 
        Auto-covariances are set to NaN.
        INPUT:
            sts: list of N spiketrains that have been annotated (exc/inh)
            binsize: quantities value for binned spiketrain
            minNspk: minimal number of spikes in a spike train (or else NaN)
        OUTPUT: 
            covm: square array N x N of cross-covariances with the constrain, 
            that each spike train in correlated pair has to consist of at least 
            minNspk spikes, otherwise assigned value is NaN. Diagonal is NaN
        '''
        binned = elephant.conversion.BinnedSpikeTrain(sts, binsize = binsize)
        covm = elephant.spike_train_correlation.covariance(binned)
        for i, st in enumerate(sts):
            if len(st) < minNspk:
                covm[i,:] = np.NaN
                covm[:,i] = np.NaN
            covm[i,i] = np.NaN
        return covm
    
    
            
    def get_ei_covar_pdf(covm, ids, 
                binrange=[-0.3, 0.3], 
                auto_cross='cross', 
                nbins=80):
        '''
        Calculates probability density function of cross-covariances.
        INPUT:
            covm: square array N x N of cross-covariances
            ids: indices of exc-exc, inh-inh, or mix-mix covariances
            binrange: binrange used for histogram
            nbins: number of bins within binrange
        OUTPUT: 
            C: dictionary of exc/inh containing elements covariance matrices
            pdf: dictionary of probability density distributions for 
                 'exc' and 'inh'
            bins: bin centers of pdf
        '''
        Nunits, _ = np.shape(covm)
        if auto_cross=='cross':
            tmp = np.copy(covm)
            np.fill_diagonal(tmp, np.nan)
        if auto_cross=='auto':
            di = np.diag_indices(Nunits)
            tmp     = np.nan
            tmp[di] = covm[di]
        C = tmp[ids,:][:,ids].ravel()
        pdf, bins = np.histogram(C, bins=nbins, 
                               range=binrange, density=True)
        bins = bins[1:]-(bins[1]-bins[0])/2
        return C, pdf, bins
    
    
    
    def get_neuron_types(sts):
        '''
        Returns list of neu_type annotations of sts
        '''
        neu_types = []
        for i in xrange(len(sts)):
            neu_types.append(sts[i].annotations['neu_type'])
        return neu_types