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
from collections import Counter

from ast import literal_eval as make_tuple

db = sqlite3.connect('results_gathered.sqlite')
db.execute('CREATE INDEX IF NOT EXISTS job_info_index ON job_info (job_id, eps, beta00, sigma01, sd_proc)')
db.execute('CREATE INDEX IF NOT EXISTS ccm_increase_index ON ccm_increase (job_id, cause, effect)')
db.execute('CREATE INDEX IF NOT EXISTS ccm_correlation_dist_index ON ccm_correlation_dist (job_id, cause, effect, L)')
db.commit()

def get_unique(pname):
    return [row[0] for row in db.execute('SELECT DISTINCT {pname} FROM job_info ORDER BY {pname}'.format(pname=pname))]

eps_vals = get_unique('eps')
assert len(eps_vals) == 2 and eps_vals[0] == 0.0

beta00_vals = get_unique('beta00')
assert len(beta00_vals) == 2

sigma01_vals = get_unique('sigma01')
sd_proc_vals = get_unique('sd_proc')

def plot_lags(eps, beta00, sigma01, sd_proc):
    job_ids = [row[0] for row in db.execute(
        'SELECT job_id FROM job_info WHERE eps = ? AND beta00 = ? AND sigma01 = ? AND sd_proc = ?', [eps, beta00, sigma01, sd_proc]
    )]
    
    max_delay = 0
    for cause, effect in [('C0', 'C1'), ('C1', 'C0')]:
        for job_id in job_ids:
            try:
                delays = make_tuple(db.execute(
                    'SELECT delays FROM ccm_increase WHERE job_id = ? AND cause = ? AND effect = ?', [job_id, cause, effect]
                ).next()[0])
                this_max_delay = max(delays)
                if this_max_delay > max_delay:
                    max_delay = this_max_delay
            except:
                pass
    
    fig = pyplot.figure(figsize=(10, 10))
    subplot_index = 0
    for cause, effect in [('C0', 'C1'), ('C1', 'C0')]:
        delays_list = []
        E_counter = Counter()
        for job_id in job_ids:
            try:
                delays = make_tuple(db.execute(
                    'SELECT delays FROM ccm_increase WHERE job_id = ? AND cause = ? AND effect = ?', [job_id, cause, effect]
                ).next()[0])
                print job_id, cause, effect, delays
                delays_list.append(delays)
                E_counter[len(delays)] += 1
            except:
                pass
        
        if len(delays_list) == 0:
            sys.stderr.write('no valid jobs for eps={}-beta00={}-sigma01={}-sd_proc={}\n'.format(eps, beta00, sigma01, sd_proc))
            return
        
        heatmap = numpy.ones((len(delays_list), max_delay + 1), dtype=float)
        for i, delays in enumerate(delays_list):
            heatmap[i, delays] = 0.0
        
        pyplot.subplot(2, 2, subplot_index*2+1)
        pyplot.imshow(heatmap, origin='upper', vmin=0, vmax=1, interpolation='none', cmap='gray')
        pyplot.xlabel('lag')
        pyplot.ylabel('replicate')
        pyplot.title('{} causes {}'.format(cause, effect))
        
        pyplot.subplot(2, 2, subplot_index*2 + 2)
        Es = sorted(E_counter.keys())
        counts = [E_counter[E] for E in Es]
        pyplot.bar([E - 0.4 for E in Es], counts, width=0.8)
        pyplot.xlabel('embedding dimension')
        pyplot.ylabel('count')
        pyplot.xticks(Es)
        pyplot.title('{} causes {}'.format(cause, effect))
        
        subplot_index += 1
    pyplot.suptitle('eps={}-beta00={}-sigma01={}-sd_proc={}'.format(eps, beta00, sigma01, sd_proc))
    
    plot_dir = os.path.join(
        'lag_plots',
        'eps={:0.1f}'.format(eps), 'beta00={:0.2f}'.format(beta00), 'sigma01={:0.2f}'.format(sigma01), 'sd_proc={:0.6f}'.format(sd_proc)
    )
    try:
        os.makedirs(plot_dir)
    except:
        pass
    pyplot.savefig(os.path.join(plot_dir, 'lag_plot.png'))
    pyplot.close(fig)

try:
    os.makedirs('lag_plots')
except:
    pass

for eps in eps_vals:
    for beta00 in beta00_vals:
        for sigma01 in sigma01_vals:
            for sd_proc in sd_proc_vals:
                plot_lags(eps, beta00, sigma01, sd_proc)

db.close()
