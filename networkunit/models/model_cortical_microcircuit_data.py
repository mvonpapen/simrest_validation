import sciunit
from networkunit import capabilities as cap
from networkunit import models

class microcircuit_data_model(models.data_model, cap.ProducesCovariances):
    """
    A model class to wrap data from already performed simulation of the
    Potjans-Diesman cortical microcircuit model.
    """
    def load(self, file_path, **kwargs):
        # ToDo: Write specific loading routine
        self.spiketrains = spiketrains
        return spiketrains

    def preprocess(self, spiketrain_list, max_subsamplesize=None, **kwargs):
        """
        Performs preprocessing on the spiketrain data according to the given
        parameters which are passed down from the test test parameters.
        """
        if spiketrain_list is not None and max_subsamplesize is not None:
            return spiketrain_list[:max_subsamplesize]
        return spiketrain_list

    def produce_covariances(self, spiketrain_list=None, **kwargs):
        """
        overwrites function in class ProduceCovariances
        """
        spiketrain_list = self.preprocess(spiketrain_list, **kwargs)
        # call generic function to calculate covariances from capability
        ProducesCovariances.produce_covariances(spiketrain_list=spiketrain_list,
                                                **kwargs)

