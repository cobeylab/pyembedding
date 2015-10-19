#!/usr/bin/env python

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
import subprocess
import multiprocessing
from datetime import datetime

sys.path.append(os.path.join(SCRIPT_DIR, 'pyembedding'))
import jsonobject

N_CORES = 4

jobs_dir = os.path.join(SCRIPT_DIR, 'jobs')
job_ids = []
for job_id_str in os.listdir(jobs_dir):
    try:
        job_id = int(job_id_str)
        job_ids.append(job_id)
    except:
        pass

job_ids = sorted(job_ids)

def run_job(job_id):
    sys.stderr.write('{0} : starting job_id {1}\n'.format(datetime.utcnow(), job_id))
    job_dir = os.path.join(jobs_dir, str(job_id))

    def write_status(status, code=None):
        status_obj = jsonobject.JSONObject([
            ('job_id', job_id),
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
            [os.path.join(SCRIPT_DIR, 'run_pipeline.py')],
            cwd=job_dir,
            stdout=stdout,
            stderr=stderr
        )
        code = proc.wait()
        sys.stderr.write(
            '{0} : job_id {1} complete with status code {2}\n'.format(datetime.utcnow(), job_id, code)
        )
        if code == 0:
            write_status('complete', code)
        else:
            write_status('failed', code)
            with open(os.path.join(job_dir, 'stderr.txt')) as f:
                stderr_data = f.read()
            sys.stderr.write('job_id {0} error:\n{1}\n'.format(job_id, stderr_data))
    except Exception as e:
        sys.stderr.write('Exception trying to run process:\n{0}\n'.format(e))

    stdout.close()
    stderr.close()

pool = multiprocessing.Pool(N_CORES)
pool.map(run_job, job_ids, chunksize=1)
