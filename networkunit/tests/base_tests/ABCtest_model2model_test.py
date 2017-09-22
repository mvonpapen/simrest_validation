import sciunit
from abc import ABCMeta, abstractmethod


class model2model_test(sciunit.Test):
    """

    """
    __metaclass__ = ABCMeta

    def __init__(self, observation, name=None, **params):
        """
        Parameters
        ----------
        observation : sciUnit.Model instance
        """
        observation = self.generate_prediction(observation, **params)
        super(model2model_test, self).__init__(observation, name=name, **params)

    @abstractmethod
    def generate_prediction(self, model, **kwargs):
        raise NotImplementedError("")
