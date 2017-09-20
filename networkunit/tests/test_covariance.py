import sciunit
from networkunit import tests
from networkunit import capabilities as cap


class covariance_two_sample_test(tests.two_sample_test):
    """
    Test to compare the pairwise covariances of a set of neurons in a network.
    The statistical testing method needs to be passed in form of a
    sciunit.Score as score_type on initialization.
    """
    required_capabilites = (cap.ProducesCovariances, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        self.params.update(kwargs)
        return model.produce_covariances(**self.params)

    def validate_observation(self, observation):
        # ToDo: Check if observation values are legit (non nan, positive, ...)
        pass

    def visualize_score(self, model, ax=None, palette=None,
                        var_name='Covariance', **kwargs):
        # sets xlabel name to 'Covariance' in visualizations
        super(covariance_two_sample_test,self).visualize_score(model, ax=ax,
                                                               palette=palette,
                                                               var_name=var_name,
                                                               **kwargs)