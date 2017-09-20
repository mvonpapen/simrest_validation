import sys
sys.path.append('../')

from quantities import ms
from networkunit.models import cortical_microcircuit_data, resting_state_data
from networkunit.scores import ks_distance
from networkunit.tests import covariance_test

"""
Validation of the microcircuit model by using data of a previously
run NEST simulation and comparing it to experimental resting state data
"""

# Loading simulation data and experimental data into data models

NEST_model_data = cortical_microcircuit_data(file_path='models/data/NEST_cort_microcircuit_model_spikes_L4I.h5',
                                             name='NEST Data Layer 4 inh')

SPINNAKER_model_data = cortical_microcircuit_data(file_path='models/data/SPINNAKER_cort_microcircuit_model_spikes_L4I.h5',
                                                  name='SpiNNaker Data Layer 4 inh')

# Initializing the test with the resting state data and setting the score type

ks_cov_test = covariance_test(reference_data=SPINNAKER_model_data,
                              name='KS Covariance Test',
                              max_subsamplesize=100,
                              align_to_0=True,
                              binsize=2 * ms,
                              t_start=0 * ms,
                              t_stop=10000 * ms,
                              data_model=True,
                              score_type=ks_distance)

# Visualize the covariances of the two data sets and plot a representation
# of the score

ks_cov_test.visualize_sample(model=NEST_model_data)

ks_cov_test.visualize_score(model=NEST_model_data)

# Perfroming the validation test against the microcircuit model

score = ks_cov_test.judge(model=NEST_model_data) # checks capabilities of model and test
                                                 # calls generate_prediction()
                                                 # calls compute_score()
                                                 # equips score with metadata
                                                 # returns score

# Printing the score outcome

score.describe()

score.summarize()

print score.summarize()
