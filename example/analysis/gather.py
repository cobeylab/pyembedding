#!/usr/bin/env python

import os
import argparse
import sys
import sqlite3

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR), 'pyembedding')
import jsonobject

parser = argparse.ArgumentParser()
parser.add_argument(
    'jobs_dir', metavar='<jobs-directory>',
    help='A directory containing output from jobs.'
)
parser.add_argument(
    '--input-db-name', metavar='<input-database-filename>',
    default='results.sqlite',
    help='The filename of the output database to be gathered from each job directory.'
)
parser.add_argument(
    '--output-db-name', metavar='<output-database-filename>',
    default='results_gathered.sqlite',
    help='The filename of the combined databases.'
)
args = parser.parse_args()

# Set up output database
output_db_path = os.path.join(args.jobs_dir, '..', args.output_db_name)
if os.path.exists(output_db_path):
    sys.stderr.write('Output database already exists; aborting.\n')
    sys.exit(1)
out_db = sqlite3.connect(output_db_path)

def load_job_db(job_dir, job_db_path):
    sys.stderr.write('{}\n'.format(job_id, job_dir))
    
    job_db = sqlite3.connect(job_db_path)
    job_id = jsonobject.load_from_file(os.path.join(job_dir, 'job_info.json')).job_id
    
    for table_name, create_sql in job_db.execute('SELECT name, sql FROM sqlite_master WHERE type = "table"'):
        # Create table if it doesn't exist
        colnames = ['job_id'] + [x[0] for x in job_db.execute('SELECT * FROM {}'.format(table_name)).description]
        out_db.execute(
            'CREATE TABLE IF NOT EXISTS {} ({})'.format(table_name, ','.join(colnames))
        )

        # Insert all rows into master database
        for row in job_db.execute('SELECT * FROM {0}'.format(table_name)):
            out_db.execute(
                'INSERT INTO {} VALUES ({})'.format(
                    table_name, ','.join(['?'] * len(colnames))
                ), [job_id] + list(row)
            )
        out_db.commit()
        
    
    job_db.close()
    

# Walk jobs directory top-down
jobs_dir = args.jobs_dir
for root, dirs, files in os.walk(jobs_dir):
    dirs.sort()
    
    job_db_path = os.path.join(root, args.input_db_name)
    if os.path.exists(job_db_path):
        load_job_db(root, job_db_path)

out_db.close()
