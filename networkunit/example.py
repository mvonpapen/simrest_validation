from networkunit.models.model_cortical_microcircuit_data import microcircuit_data_model
from networkunit.models.model_resting_state_data import resting_state_data_model
from networkunit.scores.score_ks_distance import ks_distance_score
from networkunit.tests.test_covariance import covariance_two_sample_test

"""
Validation of the microcircuit model by using data of a previously
run NEST simulation and comparing it to experimental resting state data
"""

# Loading simulation data and experimental data into data models

RS_exp_data = resting_state_data_model(file_path='filename.xx',
                                       name='RS Data v1',
                                       loadingparam='X',
                                       metadata='Y')

NEST_model_data = microcircuit_data_model(file_path='',
                                          name='NEST Data Layer 4 inh',
                                          loadingparam='XYZ')

# Initializing the test with the resting state data and setting the score type

ks_cov_test = covariance_two_sample_test(reference_data=RS_exp_data,
                                         name='KS Covariance Test',
                                         data_model=True,
                                         score_type=ks_distance_score)

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
