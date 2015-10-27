#!/usr/bin/env python

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(SCRIPT_DIR)

import sys
import subprocess

if not os.path.exists('chunks'):
    sys.stderr.write('chunks directory not present; must run generate_slurm_chunks.py first.\n')
    sys.exit(1)

def main():
    for subdir in os.listdir('chunks'):
        chunk_dir = os.path.join('chunks', subdir)
        chunk_sbatch_filename = os.path.abspath(os.path.join(chunk_dir, 'run_chunk.sbatch'))
        if os.path.exists(chunk_sbatch_filename):
            args = ['sbatch', chunk_sbatch_filename]
            print ' '.join(args)
            proc = subprocess.Popen(args, cwd=chunk_dir)
            proc.wait()

if __name__ == '__main__':
    main()
