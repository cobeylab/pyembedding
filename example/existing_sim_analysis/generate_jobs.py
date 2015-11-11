#!/usr/bin/env python

import os
import sys
import random
import sqlite3
from collections import OrderedDict
import json

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(SCRIPT_DIR, 'pyembedding'))
import uzalcost

# Simulations database: make sure this is an absolute path
#SIM_DB_PATH = '/Users/ebaskerv/uchicago/midway_cobey/ccmproject-storage/2015-10-23-simulations/results_gathered.sqlite'
SIM_DB_PATH = '/project/cobey/ccmproject-storage/2015-10-23-simulations/results_gathered.sqlite'

# Must be less than or equal to the number of replicates in the sim db (100).
# If less than 100, will only use the first N_REPLICATES replicates
#N_REPLICATES = 100
N_REPLICATES = 1

def get_ccm_settings():
    simulation_samples_per_year = 36    # This is an assumption about the time series being loaded; don't change
                                        # unless a new simulation DB is being used with different settings.
                                        # 36 implies 10-day samples in a 360-day year in the original simulations.

    timeseries_x_label = 'time (years)' # Labels for time-series plot. Rewrite to match settings
    timeseries_y_label= 'monthly cases' # below.

    years = 100                         # Number of years to use at end of time series

    ccm_samples_per_year = 12           # E.g., 12 implies 30-day samples in a 360-day year.
                                        # Must divide evenly into SIMULATION_SAMPLES_PER_YEAR

    max_theiler_window = 60             # Maximum window for exclude nearest neighbors temporally

    max_prediction_horizon = 120        # Twice the Theiler window is used to set the prediction
                                        # horizon for the Uzal cost function; it is also bounded
                                        # by this value.

    variable_name = 'C'                 # Can also be 'logS' or 'logI'.

    add_samples = True                  # If True, uses *sum* of original samples for analysis
                                        # rather than throwing out intervening samples.
                                        # For C (cases), this should probably be True, since it's a
                                        # measure of cumulative cases during a time period.

    log_transform = False               # If True, takes the natural log of samples.
                                        # If ADD_SAMPLES is True, then this is applied *after*
                                        # adding samples together.

    first_difference = False            # If True, first-differences time series.
                                        # This is applied after ADD_SAMPLES and LOG_TRANSFORM.

    standardize = False                 # If True, each time series is standardized to mean=0, sd=1.
                                        # This is applied after all other transformations.

    # Runs Uzal cost function to find upper bound on Nichkawde embedding lags, and then does bootstrapped CCM
    # at Lmin, Lmax for that embedding
    # embedding_algorithm = 'uzal_nichkawde'
    # uzal_factor = 2.0                   # Multiplies Uzal upper bound by this much
    # override_uzal_upper_bound = None    # If not None, skip Uzal algorithm and use this Nichkawde bound instead

    # Runs all valid E/tau combinations: SWEEP_EMBEDDING_DIMENSIONS x SWEEP_DELAYS
    # embedding_algorithm = 'uniform_sweep'

    # Searches for E/tau combination with highest CCM rho at Lmax, and then does bootstrapped CCM
    # at Lmin, Lmax for chosen E/tau combinations
    # embedding_algorithm = 'max_ccm_rho'

    # Searches for E/tau combination with highest univariate prediction for effect variable, and then does bootstrapped CCM
    # at Lmin, Lmax for chosen E/tau combinations
    embedding_algorithm = 'max_univariate_prediction'

    # These lists control the uniform_sweep, max_ccm_rho, and max_univariate_prediction modes above
    sweep_embedding_dimensions = range(1, 11)
    sweep_delays = [1, 2, 4]

    n_ccm_bootstraps = 1000

    return locals()

RUNMANY_INFO_TEMPLATE = OrderedDict([
    ('executable', os.path.join(SCRIPT_DIR, 'run_job.py')),
    ('minutes', 30),
    ('megabytes', 2000),
    ('simulation_db_path', SIM_DB_PATH),
    ('ccm_settings', get_ccm_settings())
])

def main():
    jobs_dir = os.path.join(SCRIPT_DIR, 'jobs')
    seed_rng = random.SystemRandom()
    
    if os.path.exists(jobs_dir):
        sys.stderr.write('{} already exists; aborting.\n'.format(jobs_dir))
        sys.exit(1)
    
    if not os.path.exists(SIM_DB_PATH):
        sys.stderr.write('simulations DB not present; aborting.\n')
        sys.exit(1)
    
    # Make sure uzal costfunc binary has been built
    sys.stderr.write('Ensuring costfunc binary has been built...\n')
    uzalcost.set_up_uzal_costfunc()
    
    with sqlite3.connect(SIM_DB_PATH) as db:
        for job_id, job_subdir, eps, beta00, sigma01, sd_proc, replicate_id, random_seed in db.execute(
            'SELECT * FROM job_info WHERE replicate_id < ?', [N_REPLICATES]
        ):
            job_dir = os.path.join(SCRIPT_DIR, job_subdir)
            sys.stderr.write('{0}\n'.format(job_dir))
            os.makedirs(job_dir)
            
            runmany_info = OrderedDict(RUNMANY_INFO_TEMPLATE)
            runmany_info['simulation_job_id'] = job_id
            runmany_info['job_info'] = OrderedDict([
                ('eps', eps),
                ('beta00', beta00),
                ('sigma01', sigma01),
                ('sd_proc', sd_proc),
                ('replicate_id', replicate_id),
                ('random_seed', seed_rng.randint(1, 2**31-1))
            ])

            dump_json(runmany_info, os.path.join(job_dir, 'runmany_info.json'))

def dump_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')

if __name__ == '__main__':
    main()
