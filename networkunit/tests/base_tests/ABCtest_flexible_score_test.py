import sciunit
from abc import ABCMeta, abstractmethod

class flexible_score_test(sciunit.Test):
    """

    """
    __metaclass__ = ABCMeta

    def __init__(self, observation, score_type, name=None, **params):
        """
        Parameters
        ----------
        score_type : sciunit.Score
            Test score to evaluate the difference of the samples of
            observation and model.
        """
        self.score_type = score_type
        super(flexible_score_test, self).__init__(observation, name=name, **params)
