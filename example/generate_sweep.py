#!/usr/bin/env python

import os
import sys
import numpy
import random
from jsonobject import JSONObject

seed_rng = random.SystemRandom()

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

beta0_vals = numpy.linspace(0.22, 0.25, num=2, endpoint=False).tolist()
sd_proc_vals = numpy.linspace(0.005, 0.01, num=2, endpoint=False).tolist()
n_replicates = 2

job_id = 0
for beta0 in beta0_vals:
    for sd_proc in sd_proc_vals:
        for replicate_id in xrange(n_replicates):
            sys.stderr.write('job_id {0}: beta0 = {1}, sd_proc = {2}, rep {3}\n'.format(job_id, beta0, sd_proc, replicate_id))
            
            job_dir = os.path.join(SCRIPT_DIR, 'jobs', '{0}'.format(job_id))
            os.makedirs(job_dir)
            
            # Write information to be used in jobs table
            # (NOTE: parameters must also be included below in format accepted by SIR simulation.)
            JSONObject([
                ('job_id', job_id),
                ('beta0', beta0),
                ('sd_proc', sd_proc),
                ('replicate_id', replicate_id),
                ('random_seed', seed_rng.randint(1, 2**31-1))
            ]).dump_to_file(os.path.join(job_dir, 'job_info.json'))
            
            # Write simulation parameters to JSON file
            JSONObject([
                ('beta0', [beta0, beta0]),
                ('sd_proc', [sd_proc, sd_proc])
            ]).dump_to_file(os.path.join(job_dir, 'sir_params.json'))
            
            # Write status 
            JSONObject([
                ('job_id', job_id),
                ('status', 'waiting'),
            ]).dump_to_file(os.path.join(job_dir, 'status.json'))
            
            job_id += 1
