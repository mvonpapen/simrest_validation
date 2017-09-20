import sciunit
import matplotlib.pyplot as plt
from networkunit.capabilities import ProducesSample
from networkunit.scores import *


class two_sample_test(sciunit.Test):
    """
    Parent class for specific two sample test scenarios which enables
    initialization via a data model instead of a direct observation,
    interchangeable test scores, and basic sample visualization.
    """
    required_capabilites = (ProducesSample, ) # Replace by more appropriate
                                              # capability in child class
                                              # i.e ProduceCovariances

    def __init__(self, reference_data, name=None, data_model=True,
                 score_type=None, **params):
        """
        The experimental data for initializing the test may be in form of a
        data structure just as the simulated model. In most cases this would
        be a list of neo.SpikeTrains for example. Therefore the input is
        transformed to the relevant observation by calling the
        generate_prediction() function, which is also applied to the model
        when the judge() function is called.

        Parameters
        ----------
        reference_data :
                The reference data can either be given in the same form as the
                model (data_model=True), or in form of the return of the
                generate_prediction() function of the test_class
                (data_model=False).
        name : str
                Name of the test. Defaults to class name.
        data_model : bool (default True)
                Whether the reference data is given in form of a model (True)
                or in form of the observation (False).
        score_type : sciunit.Score
                Test score to evaluate the difference of the samples of
                observation and model.
        params :
                Passed on to the score class.
        """
        if data_model:
            observation = self.generate_prediction(reference_data, **params)
        else:
            observation = reference_data
        self.score_type = score_type
        super(two_sample_test,self).__init__(observation, name=name, **params)

    def generate_prediction(self, model, **kwargs):
        """
        To be overwritten by child class
        """
        self.params.update(kwargs)
        try:
            return model.produce_sample(**self.params)
        except:
            raise NotImplementedError("")

    def compute_score(self, observation, prediction):
        score = self.score_type.compute(observation, prediction, **self.params)
        return score

    def visualize_sample(self, model=None, ax=None, palette=None):
        # ToDo: General visualization of a (or optionally two) data set(s)
        plt.show()
        return ax

    def visualize_score(self, model, ax=None, palette=None, **kwargs):
        """
        When there is a specific visualization function called plot() for the
        given score type, score_type.plot() is called;
        else call visualize_sample()

        Parameters
        ----------
        ax : matplotlib axis
            If no axis is passed a new figure is created.
        palette : list of color definitions
            Color definition may be a RGB sequence or a defined color code
            (i.e 'r'). Defaults to current color palette.
        Returns : matplotlib axis
        -------
        """
        try:
            self.score_type.plot(self.observation,
                                 self.generate_prediction(model),
                                 ax=ax, palette=palette, **kwargs)
        except:
            self.visualize_sample(model=model, ax=ax, palette=palette)
        return ax