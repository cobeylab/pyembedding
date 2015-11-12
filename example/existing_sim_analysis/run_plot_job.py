#!/usr/bin/env python

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(os.path.join(SCRIPT_DIR, 'pyembedding'))
import sqlite3
import numpy
import matplotlib
import random
matplotlib.use('Agg')
from matplotlib import pyplot
import json
from collections import OrderedDict

# Make sure pyembedding is set in $PYTHONPATH so these can be found, or do something like:
# sys.path.append(os.path.join(SCRIPT_DIR), 'pyembedding')
# if that's appropriate
import pyembedding
import uzalcost
import models
import statutils
import npybuffer

def main():
    '''main(): gets called at the end (after other functions have been defined)'''

    # Load job info and record in database
    if not os.path.exists('runmany_info.json'):
        sys.stderr.write('runmany_info.json missing. aborting.\n')
        sys.exit(1)
    runmany_info = load_json('runmany_info.json')
    sim_db_path = runmany_info['simulation_db_path']

    # Make sure simulation database is present
    if not os.path.exists(sim_db_path):
        sys.stderr.write('Simulation database not present; aborting\n')
        sys.exit(1)
    job_ids = runmany_info['job_ids']
    plot_settings = runmany_info['plot_settings']

    variable_name = plot_settings['variable_name']
    x0name = variable_name + '0'
    x1name = variable_name + '1'
    
    fig = pyplot.figure(figsize=(15, 5 * len(job_ids)))
    for i, job_id in enumerate(job_ids):
        X = load_simulation(sim_db_path, job_id, plot_settings)
        x0 = X[:,0]
        x1 = X[:,1]
        pyplot.subplot(len(job_ids), 1, i+1)
        plot_timeseries([x0, x1], [x0name, x1name], plot_settings['timeseries_x_label'], plot_settings['timeseries_y_label'])
    pyplot.savefig('{}.png'.format(os.path.basename(os.path.abspath('.'))))

def plot_timeseries(series, labels, xlabel, ylabel):
    for x in series:
        pyplot.plot(x)
    pyplot.legend(labels)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)

def load_simulation(sim_db_path, job_id, plot_settings):
    '''Loads and processes time series based on settings at top of file.'''
    with sqlite3.connect(sim_db_path) as sim_db:
        buf = sim_db.execute(
            'SELECT {} FROM timeseries WHERE job_id = ?'.format(plot_settings['variable_name']),
            [job_id]
        ).next()[0]
        assert isinstance(buf, buffer)
        arr = npybuffer.npy_buffer_to_ndarray(buf)
    assert arr.shape[1] == 2

    years = plot_settings['years']
    simulation_samples_per_year = plot_settings['simulation_samples_per_year']
    samples_per_year = plot_settings['samples_per_year']

    # Get the unthinned sample from the end of the time series
    sim_samps_unthinned = years * simulation_samples_per_year
    thin = simulation_samples_per_year / samples_per_year
    arr_end_unthinned = arr[-sim_samps_unthinned:, :]
    
    # Thin the samples, adding in the intervening samples if requested
    arr_mod = arr_end_unthinned[::thin, :]
    if plot_settings['add_samples']:
        for i in range(1, thin):
            arr_mod += arr_end_unthinned[i::thin, :]
    
    if plot_settings['log_transform']:
        arr_mod = numpy.log(arr_mod)
    
    if plot_settings['first_difference']:
        arr_mod = arr_mod[1:, :] - arr_mod[:-1, :]
    
    if plot_settings['standardize']:
        for i in range(arr_mod.shape[1]):
            arr_mod[:,i] -= numpy.mean(arr_mod[:,i])
            arr_mod[:,i] /= numpy.std(arr_mod[:,i])


    if plot_settings['first_difference']:
        assert arr_mod.shape[0] == years * samples_per_year - 1
    else:
        assert arr_mod.shape[0] == years * samples_per_year
    assert arr_mod.shape[1] == 2
    
    return arr_mod

def load_json(filename):
    with open(filename) as f:
        return json.load(f, object_pairs_hook=OrderedDict)

main()
