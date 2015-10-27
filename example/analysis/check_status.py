#!/usr/bin/env python

import os
import sys
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'pyembedding'))

import argparse
import time
from collections import Counter
import json
import jsonobject


parser = argparse.ArgumentParser()
parser.add_argument('-l', '-L', '--list', action='store_true', help='List status')
parser.add_argument('-s', '-S', '--status', help='Only include this status in job list')
args = parser.parse_args()

jobs_dir = os.path.join(SCRIPT_DIR, 'jobs')
counts = Counter()
for root, dirs, files in os.walk(jobs_dir):
    if not os.path.exists(os.path.join(root, 'job_info.json')):
        continue
    status_obj = None
    try:
        status_filename = os.path.join(root, 'status.json')
        if os.path.exists(status_filename):
            status_obj = jsonobject.load_from_file(status_filename)
            status = status_obj.status
        else:
            status = 'waiting'
    except Exception as e:
        sys.stderr.write('Got exception checking job in\n{}\n{}\n'.format(root, e))
        status = 'unknown'

    if args.list:
        if status_obj is None:
            status_obj = jsonobject.JSONObject([
                ('job_dir', root),
                ('status', status)
            ])
        if args.status is None or args.status == status_obj.status:
            status_obj.dump_to_file(sys.stdout)
    counts[status] += 1

json.dump(counts, sys.stdout, indent=2)
sys.stdout.write('\n')
