#!/usr/bin/env python

import os
import sys
import numpy
from jsonobject import JSONObject

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

n_ics = 2
beta0_vals = numpy.linspace(0.22, 0.25, num=2, endpoint=False).tolist()
sd_proc_vals = numpy.linspace(0.005, 0.01, num=2, endpoint=False).tolist()

def main():
    job_id = 0
    for beta0 in beta0_vals:
        for sd_proc in sd_proc_vals:
            for i in xrange(n_ics):
                sys.stderr.write('job_id {0}: beta0 = {1}, sd_proc = {2}, ic {3}\n'.format(job_id, beta0, sd_proc, i))
                
                job_dir = os.path.join(SCRIPT_DIR, 'jobs', '{0}'.format(job_id))
                os.makedirs(job_dir)
                
                JSONObject(
                    job_id = job_id
                ).dump_to_file(os.path.join(job_dir, 'job_info.json'))
                
                JSONObject(
                    beta0 = [beta0, beta0],
                    sd_proc = [sd_proc, sd_proc]
                ).dump_to_file(os.path.join(job_dir, 'sir_params.json'))
                
                JSONObject(
                    job_id = job_id,
                    status = 'waiting',
                ).dump_to_file(os.path.join(job_dir, 'status.json'))
                
                job_id += 1

if __name__ == '__main__':
    main()
