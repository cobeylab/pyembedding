#!/usr/bin/env python

import os
import sys
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(SCRIPT_DIR)

import sqlite3
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

db = sqlite3.connect('results_gathered.sqlite')
db.execute('CREATE INDEX IF NOT EXISTS job_info_index ON job_info (job_id, eps, beta00, sigma01, sd_proc)')
db.execute('CREATE INDEX IF NOT EXISTS ccm_increase_index ON ccm_increase (job_id, cause, effect)')
db.commit()

def get_unique(pname):
    return [row[0] for row in db.execute('SELECT DISTINCT {pname} FROM job_info ORDER BY {pname}'.format(pname=pname))]

eps_vals = get_unique('eps')
assert len(eps_vals) == 2 and eps_vals[0] == 0.0

beta00_vals = get_unique('beta00')
assert len(beta00_vals) == 2

sigma01_vals = get_unique('sigma01')
sd_proc_vals = get_unique('sd_proc')

def get_positive_rate(cause, effect, eps, beta00, sigma01, sd_proc):
    job_ids = [row[0] for row in db.execute(
        'SELECT job_id FROM job_info WHERE eps = ? AND beta00 = ? AND sigma01 = ? AND sd_proc = ?', [eps, beta00, sigma01, sd_proc]
    )]
    pvalues = []
    for job_id in job_ids:
        try:
            pvalue = db.execute('SELECT pvalue_increase FROM ccm_increase WHERE job_id = ? AND cause = ? AND effect = ?', [job_id, cause, effect]).next()[0]
            pvalues.append(pvalue)
        except:
            pass
    
    if len(pvalues) > 0:
        return (numpy.array(pvalues) < 0.05).sum() / float(len(pvalues))
    else:
        return float('nan')                

def plot_one(cause, effect, seasonal, different):
    heatmap = numpy.zeros((len(sigma01_vals), len(sd_proc_vals)))
    
    eps = eps_vals[seasonal]
    beta00 = beta00_vals[different]
    
    for i, sigma01 in enumerate(sigma01_vals):
        for j, sd_proc in enumerate(sd_proc_vals):
            heatmap[i,j] = get_positive_rate(cause, effect, eps, beta00, sigma01, sd_proc)
    print heatmap
    
    pyplot.xticks(range(len(sd_proc_vals)), ['{0:.2g}'.format(x) for x in sd_proc_vals])
    pyplot.xlabel('sd_proc')
    pyplot.yticks(range(len(sigma01_vals)), ['{0:.2g}'.format(y) for y in sigma01_vals])
    pyplot.ylabel('sigma01')
    
    pyplot.imshow(heatmap, origin='lower', vmin=0, vmax=1, interpolation='none')
    pyplot.colorbar()
    pyplot.title('{cause} causes {effect}'.format(cause=cause, effect=effect))

for seasonal in (0, 1):
    for different in (0, 1):
        seas_label = 'seasonal' if seasonal else 'nonseasonal'
        diff_label = 'different' if different else 'identical'
        
        fig = pyplot.figure(figsize=(12,7))
        pyplot.subplot(1, 2, 1)
        plot_one('C0', 'C1', seasonal, different)
        pyplot.subplot(1, 2, 2)
        plot_one('C1', 'C0', seasonal, different)
        pyplot.subplot(1, 2, 2)
        
        pyplot.suptitle('{}, {} - Rate of Identified Increase (fraction with p < 0.05)'.format(seas_label, diff_label))
        pyplot.savefig('positive_rate_{}_{}.png'.format(seas_label, diff_label))

db.close()
