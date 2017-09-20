"""Loads NetworkUnit model classes for NeuronUnit"""

from os.path import dirname, basename, isfile
import glob
import sciunit

"""
NOTE: All plot files must have a prefix "model_" and extension ".py".
Only these would be loaded.
"""
files = glob.glob(dirname(__file__)+"/model_*.py")
modules = [ basename(f)[:-3] for f in files if isfile(f)]

for module in modules:
    exec("from %s import *" % module)


class data_model(sciunit.Model):
    """
    A data model is representation of the experimental observation. But instead
    of containing the observation data to validate against, the data is in the
    same preprocessed form as the outcome of the model simulation.
    This requires the __init__ function of the test class to generate the
    observation data from the data_model instance.

    Minimal example of such an __init__ function:
    def __init__(self, reference_data, name=None, **params):
        observation = self.generate_prediction(reference_data)
        super(test_class_name,self).__init__(observation, name=name, **params)

    The use of a data_model enables to perfom the data analysis step more
    equivalently on both the experimental data and the simulation data.
    """
    def __init__(self, file_path, name=None, **params):
        self.data = self.load(file_path, **params)
        super(data_model, self).__init__(name=name, **params)

    def load(self, file_path, **kwargs):
        # ToDo: write generic loading routine
        data = None
        return data