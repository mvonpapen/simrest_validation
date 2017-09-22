from networkunit.tests.base_tests import covariance_test, model2model_test, flexible_score_test
from networkunit.capabilities import ProducesCovariances
from networkunit.scores import *
from abc import ABCMeta, abstractmethod

class m2m_covariance_test(covariance_test, model2model_test, flexible_score_test):
    """

    """
    test_type = 'model to model'
