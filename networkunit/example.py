import sys
sys.path.append('../')

from quantities import ms
import matplotlib.pyplot as plt
from networkunit.models import cortical_microcircuit_data
from networkunit.scores import ks_distance, best_effect_size
from networkunit.tests import covariance_test, model2model_test

"""
Validation of the microcircuit model by using data of a previously
run NEST simulation and comparing it to experimental resting state data
"""

# Loading simulation data and experimental data into data models

NEST_model_data = cortical_microcircuit_data(file_path='models/data/NEST_cort_microcircuit_model_spikes_L4I.h5',
                                             color='#FF6634',
                                             name='NEST Data Layer 4 inh')

SPINNAKER_model_data = cortical_microcircuit_data(file_path='models/data/SPINNAKER_cort_microcircuit_model_spikes_L4I.h5',
                                                  color='#1B6145',
                                                  name='SpiNNaker Data Layer 4 inh')

# Initializing the test with the resting state data and setting the score type

class m2m_cov_ks_test_2msbins_100sample(covariance_test, model2model_test):
    score_type = ks_distance
    params = {'max_subsamplesize': 100,
              'align_to_0' : True,
              'binsize' : 2 * ms,
              't_start' : 0 * ms,
              't_stop' : 10000 * ms,
              'mcmc_iter' : 10000,
              'mcmc_burn' : 100}

ks_cov_test = m2m_cov_ks_test_2msbins_100sample(observation=SPINNAKER_model_data,
                                                name='KS Covariance Test')

# Visualize the covariances of the two data sets and plot a representation
# of the score

ks_cov_test.visualize_sample(model=NEST_model_data)

# Perfroming the validation test against the microcircuit model

score = ks_cov_test.judge(model=NEST_model_data, stop_on_error=True)
# checks capabilities of model and test
# calls generate_prediction()
# calls compute_score()
# equips score with metadata
# returns score

# Printing the score outcome

ks_cov_test.visualize_score(model=NEST_model_data)

score.describe()

score.summarize()

plt.show()
