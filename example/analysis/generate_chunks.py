#!/usr/bin/env python

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(SCRIPT_DIR)

import sys
import subprocess
import multiprocessing
from datetime import datetime
import json

CHUNK_SIZE = 16
RUN_SCRIPT_FILENAME = 'run_pipeline.py'

SLURM_SCRIPT = '''#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --job-name=ns-d-{chunk_id}
#SBATCH --mem-per-cpu=2000
#SBATCH --output=stdout_slurm.txt
#SBATCH --error=stderr_slurm.txt
#SBATCH --time=0:30:00

echo SLURM_JOB_ID=$SLURM_JOB_ID
echo SLURM_JOB_NAME=$SLURM_JOB_NAME
echo SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
echo SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
echo SLURM_NODEID=$SLURM_NODEID
echo SLURM_TASK_PID=$SLURM_TASK_PID
{root_dir}/run_chunk.py
'''

if not os.path.exists('jobs'):
    sys.stderr.write('jobs does not exist; must run generate_jobs.py first.\n')
    sys.exit(1)

if os.path.exists('chunks'):
    sys.stderr.write('chunks already exists; aborting.\n')
    sys.exit(1)

def main():
    chunk_id = 0
    chunk_job_dirs = []
    for root, dirs, files in os.walk('jobs'):
        if os.path.exists(os.path.join(root, 'job_info.json')):
            chunk_job_dirs.append(os.path.abspath(root))
            if len(chunk_job_dirs) == CHUNK_SIZE:
                generate_chunk(chunk_id, chunk_job_dirs)
                chunk_id += 1
                chunk_job_dirs = []
    if len(chunk_job_dirs) > 0:
        generate_chunk(chunk_id, chunk_job_dirs)

def generate_chunk(chunk_id, chunk_job_dirs):
    chunk_dir = os.path.join('chunks', str(chunk_id))
    sys.stderr.write('{}\n'.format(chunk_dir))
    os.makedirs(chunk_dir)
    with open(os.path.join(chunk_dir, 'job_dirs.json'), 'w') as f:
        json.dump(chunk_job_dirs, f, indent=2)
        f.write('\n')
    chunk_sbatch_filename = os.path.join(chunk_dir, 'run_chunk.sbatch')
    with open(chunk_sbatch_filename, 'w') as f:
        f.write(SLURM_SCRIPT.format(
            chunk_id = chunk_id, cpus_per_task = len(chunk_job_dirs), root_dir = SCRIPT_DIR
        ))

main()
