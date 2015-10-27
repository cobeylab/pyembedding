#!/usr/bin/env python

# Make sure this is an absolute path
#SIM_DB_PATH = '/Users/ebaskerv/uchicago/midway_cobey/2015-10-23-simulations/results_gathered.sqlite'
SIM_DB_PATH = '/project/cobey/ccmproject-storage/2015-10-23-simulations/results_gathered.sqlite'

# Must be less than or equal to the number of replicates in the sim db (100).
# If less than 100, will only use the first N_REPLICATES replicates
#N_REPLICATES = 100
N_REPLICATES = 1

import os
import sys
import numpy
import sqlite3

# Put jobs in the subdirectory of directory containing this script
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(SCRIPT_DIR)

sys.path.append(os.path.join(SCRIPT_DIR, 'pyembedding'))
from jsonobject import JSONObject

# Set up system entropy source as generator for individual CCM analysis seeds
import random
seed_rng = random.SystemRandom()

if os.path.exists('jobs'):
    sys.stderr.write('jobs directory already exists; aborting.\n')
    sys.exit(1)

if not os.path.exists(SIM_DB_PATH):
    sys.stderr.write('simulations DB not present; aborting.\n')
    sys.exit(1)

with sqlite3.connect(SIM_DB_PATH) as db:
    for job_id, job_dir, eps, beta00, sigma01, sd_proc, replicate_id, sim_random_seed in db.execute(
        'SELECT * FROM job_info WHERE replicate_id < ?', [N_REPLICATES]
    ):
        sys.stderr.write('{0}\n'.format(job_dir))
        os.makedirs(job_dir)
    
        # Write information to be used in jobs table
        JSONObject([
            ('job_id', job_id),
            ('job_dir', job_dir),
            ('eps', eps),
            ('beta00', beta00),
            ('sigma01', sigma01),
            ('sd_proc', sd_proc),
            ('replicate_id', replicate_id),
            ('random_seed', seed_rng.randint(1, 2**31-1))
        ]).dump_to_file(os.path.join(job_dir, 'job_info.json'))
