#!/usr/bin/env python

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(os.path.join(SCRIPT_DIR, 'pyembedding'))

import subprocess
import multiprocessing
from datetime import datetime
import json
import jsonobject

RUN_SCRIPT_FILENAME = 'run_job.py'

jobs_dir = os.path.join(SCRIPT_DIR, 'jobs')
with open('job_dirs.json') as f:
    job_dirs = json.load(f)
n_jobs = len(job_dirs)

def run_job(job_dir):
    sys.stderr.write('{0} : starting job:\n  {1}\n'.format(datetime.utcnow(), job_dir))

    def write_status(status, code=None):
        status_obj = jsonobject.JSONObject([
            ('job_dir', job_dir),
            ('status', status)
        ])
        if code is not None:
            status_obj.code = code
        status_obj.dump_to_file(os.path.join(job_dir, 'status.json'))

    stdout = open(os.path.join(job_dir, 'stdout.txt'), 'w')
    stderr = open(os.path.join(job_dir, 'stderr.txt'), 'w')

    write_status('running')
    try:
        proc = subprocess.Popen(
            [os.path.join(SCRIPT_DIR, RUN_SCRIPT_FILENAME)],
            cwd=job_dir,
            stdout=stdout,
            stderr=stderr
        )
        code = proc.wait()
        sys.stderr.write(
            '{} : job complete with status code {}:\n  {}\n'.format(datetime.utcnow(), code, job_dir)
        )
        if code == 0:
            write_status('complete', code)
        else:
            write_status('failed', code)
            with open(os.path.join(job_dir, 'stderr.txt')) as f:
                stderr_data = f.read()
            sys.stderr.write('error:\n{}\n{}\n'.format(job_dir, stderr_data))
    except Exception as e:
        sys.stderr.write('Exception trying to run process:\n{0}\n'.format(e))

    stdout.close()
    stderr.close()

pool = multiprocessing.Pool(n_jobs)
pool.map(run_job, job_dirs, chunksize=1)
