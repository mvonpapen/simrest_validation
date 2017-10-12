import sciunit
from networkunit.capabilities import ProducesSpikeTrains
import numpy as np
from elephant.spike_train_generation import single_interaction_process as SIP
from elephant.spike_train_generation import compound_poisson_process as CPP
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from quantities import ms, Hz, quantity
import neo
import random


class stochastic_activity(sciunit.Model, ProducesSpikeTrains):
    """

    """
    params = {'size': 100,
              't_start': 0 * ms,
              't_stop': 10000 * ms,
              'rate': 10 * Hz,
              'statistic': 'poisson',
              'correlation_method': 'CPP', # 'spatio-temporal'
              'expected_binsize': 2 * ms,
              'correlations': 0.,
              'assembly_sizes': [],
              'bkgr_correlation': 0.,
              'max_pattern_length':100 * ms,
              'shuffle': False,
              'shuffle_seed': None}

    def __init__(self, name=None, **params):
        self.params.update(params)
        # updating params is only for testing reasons
        # for usage in the validation framework, the params need to be fixed!
        self.__dict__.update(self.params)
        self.check_input()
        self.spiketrains = self.generate_spiketrains()
        super(stochastic_activity, self).__init__(name=name, **self.params)

    def check_input(self):
        if not type(self.correlations) == list:
            self.correlations = [self.correlations] * len(self.assembly_sizes)
        elif len(self.correlations) == 1:
            self.correlations *= len(self.assembly_sizes)
        pass

    def produce_spiketrains(self):
        return self.spiketrains

    def generate_spiketrains(self, **kwargs):
        spiketrains = [None] * self.size

        if self.correlation_method == 'pairwise_equivalent':
        # change input to pairwise correlations with expected distribution
        # correlation coefficients
            nbr_of_pairs = [0] * len(self.assembly_sizes)
            new_correlation = []
            for i, A_size in enumerate(self.assembly_sizes):
                nbr_of_pairs[i] = A_size * (A_size - 1) / 2
                new_correlation = new_correlation + [self.correlations[i]]*nbr_of_pairs[i]
            if sum(nbr_of_pairs)*2 > self.size:
                raise ValueError, 'Assemblies are too large to generate an ' \
                                  'pairwise equivalent with the network size.'
            print nbr_of_pairs
            self.assembly_sizes = [2] * sum(nbr_of_pairs)
            self.correlations = new_correlation
            self.correlation_method = 'CPP'

        # generate correlated assemblies
        for i, a_size in enumerate(self.assembly_sizes):
            if a_size < 2:
                raise ValueError, 'An assembly must consists of at least two units.'
            generated_sts = int(np.sum(self.assembly_sizes[:i]))
            spiketrains[generated_sts:generated_sts + a_size] \
                = self._generate_assembly(correlation=self.correlations[i],
                                          A_size=a_size)
            for j in range(a_size):
                spiketrains[generated_sts + j].annotations = {'Assembly {}'.format(i)}

        # generate background
        if self.bkgr_correlation > 0:
            dummy = None
            # ToDo: background generation without cpp
        else:
            spiketrains[sum(self.assembly_sizes):] \
                = np.array([HPP(rate=self.rate, t_start=self.t_start, t_stop=self.t_stop)
                            for _ in range(self.size - sum(self.assembly_sizes))])

        if self.shuffle:
            if self.shuffle_seed is None:
                random.shuffle(spiketrains)
            else:
                random.Random(self.shuffle_seed).shuffle(spiketrains)

        return spiketrains

    def _generate_assembly(self, correlation, A_size, **kwargs):

        syncprob = self._correlation_to_syncprob(cc=correlation,
                                                 A_size=A_size,
                                                 rate=self.rate,
                                                 T=self.t_stop - self.t_start,
                                                 binsize=self.expected_binsize)
        bkgr_syncprob = self._correlation_to_syncprob(cc=self.bkgr_correlation,
                                                      A_size=2,
                                                      rate=self.rate,
                                                      T=self.t_stop - self.t_start,
                                                      binsize=self.expected_binsize)
        if self.correlation_method == 'CPP' \
        or self.correlation_method == 'spatio_temporal':
            assembly_sts = self._generate_CPP_assembly(A_size=A_size,
                                                       syncprob=syncprob,
                                                       bkgr_syncprob=bkgr_syncprob)
            if self.correlation_method == 'CPP':
                return assembly_sts
            else:
                return self._shift_spiketrains(assembly_sts)
        else:
            raise NameError("Method name not known!")

    def _generate_CPP_assembly(self, A_size, syncprob, bkgr_syncprob):
        amp_dist = np.zeros(A_size + 1)
        amp_dist[1] = 1. - syncprob - bkgr_syncprob
        amp_dist[2] = bkgr_syncprob
        amp_dist[A_size] = syncprob
        np.testing.assert_almost_equal(sum(amp_dist), 1., decimal=1)
        amp_dist *= (1. / sum(amp_dist))
        return CPP(rate=self.rate, A=amp_dist,
                   t_start=self.t_start, t_stop=self.t_stop)

    def _shift_spiketrains(self, assembly_sts):#
        shifted_assembly_sts = [None] * len(assembly_sts)
        for i, st in enumerate(assembly_sts):
            spiketimes = np.array(st.tolist())
            shift = np.random.rand() * self.max_pattern_length \
                                     - self.max_pattern_length
            shift = float(shift.rescale('ms'))
            pos_fugitives = np.where(spiketimes + shift >= float(self.t_stop))[0]
            neg_fugitives = np.where(spiketimes + shift <= float(self.t_start))[0]
            spiketimes[pos_fugitives] = spiketimes[pos_fugitives] - float(
                self.t_stop)
            spiketimes[neg_fugitives] = spiketimes[neg_fugitives] + float(
                self.t_stop - self.t_start)
            shifted_assembly_sts[i] = neo.SpikeTrain(times=spiketimes,
                                                     units='ms',
                                                     t_start=self.t_start,
                                                     t_stop=self.t_stop)
        return shifted_assembly_sts

    def _correlation_to_syncprob(self, cc, A_size, rate, T, binsize):
        if A_size < 2:
            raise ValueError
        if cc == 1:
            return 1
        m0 = rate * T / (float(T)/float(binsize))
        if type(m0) == quantity.Quantity:
            m0 = m0.rescale('dimensionless')
        n = float(A_size)

        root = np.sqrt(cc ** 2 * n ** 2
                       - 2 * cc ** 2 * n
                       + cc ** 2
                       + 4 * cc * m0 * n
                       - 4 * cc * m0
                       - 2 * cc * n ** 2
                       + 2 * cc * n
                       - 4 * m0 * n
                       + 4 * m0
                       + n ** 2)

        adding = (- 2 * cc * m0 * n
                  + 2 * cc * m0
                  + cc * n ** 2
                  - cc * n
                  + 2 * m0 * n
                  - 2 * m0
                  - n ** 2)

        denominator = 2 * (cc - 1.) * m0 * (n - 1.) ** 2

        sync_prob = (n * root + adding) / denominator

        if type(sync_prob) == quantity.Quantity:
            if bool(sync_prob.dimensionality):
                raise ValueError
            else:
                return float(sync_prob.magnitude)
        else:
            return sync_prob

    # def show_rasterplot(self, **kawrgs):
    #     ToDo: include viziphant rasterplot



# Todo: Handle quantitiy inputs which are not ms or Hz
