"""Loads NetworkUnit capability classes for NeuronUnit"""

from os.path import dirname, basename, isfile
import glob
import sciunit

"""
NOTE: All capability files must have a prefix "cap_" and extension ".py".
Only these would be loaded.
"""
files = glob.glob(dirname(__file__)+"/cap_*.py")
modules = [ basename(f)[:-3] for f in files if isfile(f)]

for module in modules:
    exec("from {} import *".format(module))


class ProducesSample(sciunit.Capability):
    """
    Here general porperties and checks of capabilities can be defined which
    produce a sample of a property.
    """

    def produce_sample(self):
        return self.unimplemented()