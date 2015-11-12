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

def get_increase_positive_rate(cause, effect, eps, beta00, sigma01, sd_proc):
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

def get_mean_pvalue(cause, effect, eps, beta00, sigma01, sd_proc):
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
        return numpy.mean(pvalues)
    else:
        return float('nan')

def get_positive_positive_rate(cause, effect, eps, beta00, sigma01, sd_proc):
    job_ids = [row[0] for row in db.execute(
        'SELECT job_id FROM job_info WHERE eps = ? AND beta00 = ? AND sigma01 = ? AND sd_proc = ?', [eps, beta00, sigma01, sd_proc]
    )]
    pvalues = []
    for job_id in job_ids:
        try:
            Lmax = db.execute(
                'SELECT MAX(L) FROM ccm_correlation_dist WHERE job_id = ? AND cause = ? AND effect = ?',
                [job_id, cause, effect]
            ).next()[0]
            pvalue = db.execute(
                'SELECT pvalue_positive FROM ccm_correlation_dist WHERE job_id = ? AND cause = ? AND effect = ? AND L = ?',
                [job_id, cause, effect, Lmax]
            ).next()[0]
            pvalues.append(pvalue)
        except:
            pass
    
    if len(pvalues) > 0:
        return (numpy.array(pvalues) < 0.05).sum() / float(len(pvalues))
    else:
        return float('nan')

def get_mean_rho(cause, effect, eps, beta00, sigma01, sd_proc):
    job_ids = [row[0] for row in db.execute(
        'SELECT job_id FROM job_info WHERE eps = ? AND beta00 = ? AND sigma01 = ? AND sd_proc = ?', [eps, beta00, sigma01, sd_proc]
    )]
    corrs = []
    for job_id in job_ids:
        try:
            Lmax = db.execute(
                'SELECT MAX(L) FROM ccm_correlation_dist WHERE job_id = ? AND cause = ? AND effect = ?',
                [job_id, cause, effect]
            ).next()[0]
            corr = db.execute(
                'SELECT mean FROM ccm_correlation_dist WHERE job_id = ? AND cause = ? AND effect = ? AND L = ?',
                [job_id, cause, effect, Lmax]
            ).next()[0]
            corrs.append(corr)
        except:
            pass
    
    return numpy.mean(corrs)

def plot_heatmap(mat, xlabel, xticks, ylabel, yticks, vmin=0, vmax=1):
    pyplot.imshow(mat, origin='lower', vmin=vmin, vmax=vmax, interpolation='none')
    
    pyplot.xticks(range(mat.shape[1]), xticks)
    pyplot.xlabel(xlabel)
    pyplot.yticks(range(mat.shape[0]), yticks)
    pyplot.ylabel(ylabel)
    
    pyplot.colorbar()

def plot_func(cause, effect, seasonal, different, heatmap_func, title):
    heatmap = numpy.zeros((len(sigma01_vals), len(sd_proc_vals)))
    
    eps = eps_vals[seasonal]
    beta00 = beta00_vals[different]
    
    for i, sigma01 in enumerate(sigma01_vals):
        for j, sd_proc in enumerate(sd_proc_vals):
            heatmap[i,j] = heatmap_func(cause, effect, eps, beta00, sigma01, sd_proc)
    print heatmap
    
    plot_heatmap(heatmap,
        'sd_proc', ['{0:.2g}'.format(x) for x in sd_proc_vals],
        'sigma01', ['{0:.2g}'.format(y) for y in sigma01_vals],
        vmin=0, vmax=1
    )
    pyplot.title('{}: {} causes {}'.format(title, cause, effect))

def plot_increase_positive_rate(cause, effect, seasonal, different):
    plot_func(cause, effect, seasonal, different, get_increase_positive_rate, 'frac(pval increase < 0.05)')

def plot_mean_pvalue(cause, effect, seasonal, different):
    plot_func(cause, effect, seasonal, different, get_mean_pvalue, 'mean(pval increase)')

def plot_positive_positive_rate(cause, effect, seasonal, different):
    plot_func(cause, effect, seasonal, different, get_positive_positive_rate, 'frac(pval pos at Lmax < 0.05)')

def plot_mean_rho(cause, effect, seasonal, different):
    plot_func(cause, effect, seasonal, different, get_mean_rho, 'mean ccm rho at Lmax')

for seasonal in (0, 1):
    for different in (0, 1):
        seas_label = 'seasonal' if seasonal else 'nonseasonal'
        diff_label = 'different' if different else 'identical'
        
        fig = pyplot.figure(figsize=(12,21))
        pyplot.subplot(4, 2, 1)
        plot_increase_positive_rate('C0', 'C1', seasonal, different)
        pyplot.subplot(4, 2, 2)
        plot_increase_positive_rate('C1', 'C0', seasonal, different)
        
        pyplot.subplot(4, 2, 3)
        plot_mean_pvalue('C0', 'C1', seasonal, different)
        pyplot.subplot(4, 2, 4)
        plot_mean_pvalue('C1', 'C0', seasonal, different)
        
        pyplot.subplot(4, 2, 5)
        plot_mean_rho('C0', 'C1', seasonal, different)
        pyplot.subplot(4, 2, 6)
        plot_mean_rho('C1', 'C0', seasonal, different)
        
        pyplot.subplot(4, 2, 7)
        plot_positive_positive_rate('C0', 'C1', seasonal, different)
        pyplot.subplot(4, 2, 8)
        plot_positive_positive_rate('C1', 'C0', seasonal, different)
        
        pyplot.suptitle('{}, {}'.format(seas_label, diff_label))
        pyplot.savefig('positive_rate_{}_{}.png'.format(seas_label, diff_label))

db.close()
