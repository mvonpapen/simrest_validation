import sciunit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from networkunit.tests.base_tests.ABCtest_two_sample_test import two_sample_test
from networkunit.capabilities import ProducesCovariances
from abc import ABCMeta, abstractmethod


class covariance_test(two_sample_test):
    """
    Test to compare the pairwise covariances of a set of neurons in a network.
    The statistical testing method needs to be passed in form of a
    sciunit.Score as score_type on initialization.
    """
    __metaclass__ = ABCMeta

    required_capabilities = (ProducesCovariances, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        return model.produce_covariances(**self.params)

    def validate_observation(self, observation):
        # ToDo: Check if observation values are legit (non nan, positive, ...)
        pass

    def visualize_score(self, model, ax=None, palette=None,
                        var_name='Covariance',
                        **kwargs):
        # sets xlabel name to 'Covariance' in visualizations
        super(covariance_test,self).visualize_score(model, ax=ax,
                                                    palette=palette,
                                                    var_name=var_name,
                                                    **kwargs)

    def visualize_sample(self, model=None, ax=None, bins=100, palette=None,
                         sample_names=['observation', 'prediction'],
                         var_name='Covariance', **kwargs):
        # sets xlabel name to 'Covariance' in visualizations
        super(covariance_test,self).visualize_sample(model=model, ax=ax,
                                                     palette=palette,
                                                     bins=bins,
                                                     sample_names=sample_names,
                                                     var_name=var_name,
                                                     **kwargs)