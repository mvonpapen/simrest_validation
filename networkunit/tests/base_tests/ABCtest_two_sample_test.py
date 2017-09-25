import sciunit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from abc import ABCMeta, abstractmethod


class two_sample_test(sciunit.Test):
    """
    Parent class for specific two sample test scenarios which enables
    initialization via a data model instead of a direct observation,
    interchangeable test scores, and basic sample visualization.
    """
    __metaclass__ = ABCMeta

    # required_capabilites = (ProducesSample, ) # Replace by more appropriate
                                              # capability in child class
                                              # i.e ProduceCovariances

    def generate_prediction(self, model, **kwargs):
        """
        To be overwritten by child class
        """
        self.params.update(kwargs)
        try:
            return model.produce_sample(**self.params)
        except:
            raise NotImplementedError("")

    def compute_score(self, observation, prediction, **kwargs):
        self.params.update(kwargs)
        score = self.score_type.compute(observation, prediction, **self.params)
        return score

    def visualize_sample(self, model=None, ax=None, bins=100, palette=None,
                         sample_names=['observation', 'prediction'],
                         var_name='Measured Parameter', **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]
        if model is None:
            sample2 = None
        else:
            sample2 = self.generate_prediction(model, **self.params)

        sample1 = self.observation

        if model is None:
            P, edges = np.histogram(sample1, bins=bins, density=True)
            ymax = max(P)
        else:
            if np.amax(sample1) >= np.amax(sample2):
                P, edges = np.histogram(sample1, bins=bins, density=True)
                Q, _____ = np.histogram(sample2, bins=edges, density=True)
            else:
                Q, edges = np.histogram(sample2, bins=bins, density=True)
                P, _____ = np.histogram(sample1, bins=edges, density=True)
            ymax = max(max(P), max(Q))
            Q = np.append(np.append(0., Q), 0.)

        P = np.append(np.append(0., P), 0.)
        dx = np.diff(edges)[0]
        xvalues = edges[:-1] + dx / 2.
        xvalues = np.append(np.append(xvalues[0] - dx, xvalues),
                            xvalues[-1] + dx)
        ax.plot(xvalues, P, label=sample_names[0], color=palette[0])
        if model is not None:
            ax.plot(xvalues, Q, label=sample_names[1], color=palette[1])
        ax.set_xlim(xvalues[0], xvalues[-1])
        ax.set_ylim(0, ymax)
        ax.set_ylabel('Density')
        ax.set_xlabel('Measured Parameter')
        plt.legend()
        # plt.show()
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