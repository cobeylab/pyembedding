#!/usr/bin/env python

import os
import sys
import argparse
import time
from collections import Counter
import json
import jsonobject

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('-l', '-L', '--list', action='store_true', help='List status')
parser.add_argument('-s', '-S', '--status', help='Only include this status in job list')
args = parser.parse_args()

jobs_dir = os.path.join(SCRIPT_DIR, 'jobs')
job_ids = []
for job_id_str in os.listdir(jobs_dir):
    try:
        job_id = int(job_id_str)
        job_ids.append(job_id)
    except:
        pass

counts = Counter()
for job_id in sorted(job_ids):
    status_obj = None

    try:
        status_filename = os.path.join(SCRIPT_DIR, 'jobs', str(job_id), 'status.json')
        if os.path.exists(status_filename):
            status_obj = jsonobject.load_from_file(status_filename)
            status = status_obj.status
        else:
            status = 'waiting'
    except Exception as e:
        sys.stderr.write('Got exception checking job id {0}:\n{1}\n'.format(e))
        status = 'unknown'

    if args.list:
        if status_obj is None:
            status_obj = jsonobject.JSONObject([
                ('job_id', job_id),
                ('status', status)
            ])
        if args.status is None or args.status == status_obj.status:
            status_obj.dump_to_file(sys.stdout)
    counts[status] += 1

json.dump(counts, sys.stdout, indent=2)
sys.stdout.write('\n')
