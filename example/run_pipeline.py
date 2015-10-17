#!/usr/bin/env python
'''Runs a single simulation and CCM analysis.'''

import os
import sys
import subprocess
import multiprocessing
import numpy
import pyembedding
import jsonobject
import models

def main():
    # Initialize base parameters
    params = jsonobject.JSONObject({
        'n_pathogens' : 2,

        'dt_euler' : 0.01,
        'adaptive' : False,
        'tol' : 1e-6,
        't_end' : 72000.0,
        'dt_output' : 360.0,

        'mu' : 0.00054794520548,
        'nu' : [0.2, 0.2],
        'gamma' : [0.0, 0.0],

        'beta0' : [0.22, 0.22],
        'S_init' : [0.5, 0.5],
        'I_init' : [0.05, 0.05],
        'beta_change_start' : [0.0, 0.0],
        'beta_slope' : [0.0, 0.0],
        'psi' : [360.0, 360.0],
        'omega' : [0.0, 0.0],
        'eps' : [0.1, 0.1],
        'sigma' : [[1.0, 0.01], [0.0, 1.0]],

        'shared_proc' : False,
        'sd_proc' : [0.00, 0.00],

        'shared_obs' : False,
        'sd_obs' : [0.0, 0.0],

        'shared_obs_C' : False,
        'sd_obs_C' : [0.0, 0.0]
    })

    # If this is a sweep: load extra parameters from parameters.json file
    # These override base parameters
    #with open('parameters.json') as f:
    #    params.load_from_file(f)

    # Run simulation
    try:
        sir_out = models.run_via_pypy('multistrain_sde', params)
    except models.ExecutionException as e:
        sys.stderr.write('{0}\n'.format(e.cause))
        sys.stderr.write(e.stderr_data)
        return
    #x = numpy.random.normal(0, 1, size=1200)
    #y = numpy.random.normal(0, 1, size=1200)

    print sir_out.C
    C0 = sir_out.C[-100:,0]
    C1 = sir_out.C[-100:,1]

    # Identify embedding for each variable
    C0_embedding = pyembedding.Embedding(C0, (0,1))
    C1_embedding = pyembedding.Embedding(C1, (0,1))

    # Run CCM in both directions
    for i in range(1000):
        C1_subsamp = C1_embedding.sampled_embedding(100)
        ccm_result, y_actual, y_pred = C1_subsamp.ccm(C1_embedding, C0, theiler_window=10)
        print ccm_result.dump_to_string()

    # embedding_params = choose_embedding_parameters(x_embedding, y_embedding)

# def run_simulation():
#
#
# def identify_embedding(x):
#     '''Identify best univariate embedding for a time series.
#
#     (1) Calculates Theiler window and maximum prediction horizon.
#     (2) Identifies best full embedding from Uzal Lk cost function.
#     (3) Identifies best sub-embedding using Nichkawde MDOP method.
#     '''
#     theiler_window = 3 * pyembedding.autocorrelation_threshold_delay(x, 1.0/numpy.e)
#     prediction_window = 2 * theiler_window
#
#     uzal_full_embedding = pyembedding.uzal_full_embedding(x)
#
# def choose_embedding_parameters(e1, e2):
#     # Decide on embedding t

if __name__ == '__main__':
    main()
