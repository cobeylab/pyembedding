#!/usr/bin/env python

# Make sure this is an absolute path
SIM_DB_PATH = '/Users/ebaskerv/uchicago/midway_cobey/2015-10-23-simulations/results_gathered.sqlite'
#SIM_DB_PATH = '/project/cobey/ccmprojet-storage/2015-10-23-simulations/results_gathered.sqlite'

SIMULATION_SAMPLES_PER_YEAR = 36    # This is an assumption about the time series being loaded; don't change
                                    # unless a new simulation DB is being used with different settings.
                                    # 36 implies 10-day samples in a 360-day year in the original simulations.

CCM_YEARS = 100                     # Number of years to use at end of time series

CCM_SAMPLES_PER_YEAR = 12           # E.g., 12 implies 30-day samples in a 360-day year.
                                    # Must divide evenly into SIMULATION_SAMPLES_PER_YEAR

VARIABLE_NAME = 'C'                 # Can also be 'logS' or 'logI'.

ADD_SAMPLES = True                  # If True, uses *sum* of original samples for analysis
                                    # rather than throwing out intervening samples.
                                    # For C (cases), this should probably be True, since it's a
                                    # measure of cumulative cases during a time period.

LOG_TRANSFORM = False               # If True, takes the natural log of samples.
                                    # If ADD_SAMPLES is True, then this is applied *after*
                                    # adding samples together.

FIRST_DIFFERENCE = False            # If True, first-differences time series.
                                    # This is applied after ADD_SAMPLES and LOG_TRANSFORM.

EMBEDDING_ALGORITHM = 'uzal_nichkawde'

N_CCM_BOOTSTRAPS = 1000

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
import jsonobject
import models
import statutils
import npybuffer

# Make sure simulation database is present
if not os.path.exists(SIM_DB_PATH):
    sys.stderr.write('Simulation database not present; aborting\n')
    sys.exit(1)

# Connect to output database
if os.path.exists('results.sqlite'):
    sys.stderr.write('Output database present. Aborting.\n')
    sys.exit(1)
db = sqlite3.connect('results.sqlite')

# Load job info and record in database
if not os.path.exists('job_info.json'):
    sys.stderr.write('job_info.json missing. aborting.\n')
    sys.exit(1)
job_info = jsonobject.load_from_file('job_info.json')
job_id = job_info.job_id
db.execute('CREATE TABLE job_info ({0})'.format(', '.join([key for key in job_info.keys()])))
db.execute('INSERT INTO job_info VALUES ({0})'.format(', '.join(['?'] * len(job_info))), job_info.values())

# Set up RNG
rng = numpy.random.RandomState(job_info.random_seed)

def main():
    '''main(): gets called at the end (after other functions have been defined)'''
    X = load_simulation()
    x0 = X[:,0]
    x1 = X[:,1]
    
    run_analysis(VARIABLE_NAME + '0', x0, VARIABLE_NAME + '1', x1)
    run_analysis(VARIABLE_NAME + '1', x1, VARIABLE_NAME + '0', x0)

    db.commit()
    db.close()

def load_simulation():
    '''Loads and processes time series based on settings at top of file.'''
    with sqlite3.connect(SIM_DB_PATH) as sim_db:
        buf = sim_db.execute(
            'SELECT {} FROM timeseries WHERE job_id = ?'.format(VARIABLE_NAME),
            [job_id]
        ).next()[0]
        assert isinstance(buf, buffer)
        arr = npybuffer.npy_buffer_to_ndarray(buf)
    assert arr.shape[1] == 2
    
    # Get the unthinned sample from the end of the time series
    sim_samps_unthinned = CCM_YEARS * SIMULATION_SAMPLES_PER_YEAR
    thin = SIMULATION_SAMPLES_PER_YEAR / CCM_SAMPLES_PER_YEAR
    arr_end_unthinned = arr[-sim_samps_unthinned:, :]
    
    # Thin the samples, adding in the intervening samples if requested
    arr_mod = arr_end_unthinned[::thin, :]
    if ADD_SAMPLES:
        for i in range(1, thin):
            arr_mod += arr_end_unthinned[i::thin, :]
    
    if LOG_TRANSFORM:
        arr_mod = numpy.log(arr_mod)
    
    if FIRST_DIFFERENCE:
        arr_mod = arr_mod[1:, :] - arr_mod[:-1, :]
        assert arr_mod.shape[0] == CCM_YEARS * CCM_SAMPLES_PER_YEAR - 1
    else:
        assert arr_mod.shape[0] == CCM_YEARS * CCM_SAMPLES_PER_YEAR
    assert arr_mod.shape[1] == 2
    
    return arr_mod

def run_analysis(cname, cause, ename, effect):
    '''Run analysis for a single causal direction.'''
    sys.stderr.write('Running {0}-causes-{1}\n'.format(cname, ename))

    # Check if effect has no variation
    cause_sd = numpy.std(cause)
    effect_sd = numpy.std(effect)
    if cause_sd == 0.0 and effect_sd == 0.0:
        sys.stderr.write('No variation in either time series; skipping analysis.\n')
        return
    elif effect_sd == 0.0:
        sys.stderr.write('Warning: no variation in effect time series. Using hard-coded Theiler window = 0, emb. dim. = 1')
        theiler_window = 0
        max_embedding_dimension = 1
    else:
        # Identify delay at which autocorrelation drops to 1/e
        ac_delay, autocorr = pyembedding.autocorrelation_threshold_delay(effect, 1.0/numpy.e)
        sys.stderr.write('  ac_delay, autocorr = {0}, {1}\n'.format(ac_delay, autocorr))

        # Calculate Theiler window (limit on closeness of neighbors in time)
        theiler_window = 3 * ac_delay
        sys.stderr.write('  theiler_window = {0}\n'.format(theiler_window))
        assert theiler_window < effect.shape[0]

        # Calculate maximum prediction horizon (used by Uzal cost function)
        prediction_horizon = 2 * theiler_window
        sys.stderr.write('  prediction_horizon = {0}\n'.format(prediction_horizon))
        assert prediction_horizon > theiler_window

        # Hard-code maximum delay vector window:, equal to delay * (embedding_dimension - 1)
        max_window = 60
        sys.stderr.write('  max_window = {0}\n'.format(max_window))

        # Run Uzal cost function (will implicitly compile Uzal's C code if necessary)
        ms, Lks, params = uzalcost.run_uzal_costfunc(
            effect, theiler_window=theiler_window, max_prediction_horizon=prediction_horizon,
            max_window=max_window
        )
        best_m_index = numpy.argmin(Lks)
        max_embedding_dimension = ms[best_m_index]
        Lk = Lks[best_m_index]
        sys.stderr.write('  Uzal full embedding dimension = {0} (Lk = {1})\n'.format(max_embedding_dimension, Lk))

    # Run Nichkawde algorithm to identify sub-embedding
    embedding, derivs_tup, fnn_rates_tup = pyembedding.nichkawde_embedding(effect, theiler_window, max_embedding_dimension, return_metrics=True)
    delays = embedding.delays
    write_and_plot_nichkawde_metrics(cname, ename, delays, derivs_tup, fnn_rates_tup)

    sys.stderr.write('  Nichkawde sub-embedding: {0}\n'.format(delays))

    # min library size: embedding_dimension + 2,
    # so vectors should usually have embedding_dimension + 1 neighbors available
    Lmin = embedding.embedding_dimension + 2

    # max library size: just the number of available delay vectors
    Lmax = embedding.delay_vector_count

    sys.stderr.write('  Using Lmin = {0}, Lmax = {1}\n'.format(Lmin, Lmax))

    assert Lmax > Lmin

    corrs_Lmin = run_ccm_bootstraps(cname, ename, embedding, cause, Lmin, theiler_window)
    corrs_Lmax = run_ccm_bootstraps(cname, ename, embedding, cause, Lmax, theiler_window)

    db.execute(
        'CREATE TABLE IF NOT EXISTS ccm_increase (cause, effect, Lmin, Lmax, delays, pvalue_increase)'
    )
    db.execute(
        'INSERT INTO ccm_increase VALUES (?,?,?,?,?,?)',
        [cname, ename, Lmin, Lmax, str(embedding.delays), 1.0 - numpy.mean(statutils.inverse_quantile(corrs_Lmin, corrs_Lmax))]
    )
    db.commit()

def write_and_plot_nichkawde_metrics(cname, ename, delays, derivs_tup, fnn_rates_tup):
    fig = pyplot.figure(figsize=(10, 5*len(derivs_tup)))
    
    db.execute('CREATE TABLE IF NOT EXISTS nichkawde_metrics (delays, geo_mean_derivs, fnn_rates)')
    
    for i in range(len(derivs_tup)):
        derivs = derivs_tup[i]
        fnn_rates = fnn_rates_tup[i]
        
        db.execute('INSERT INTO nichkawde_metrics VALUES (?,?,?)', [
            json.dumps(delays[:(i+1)]),
            json.dumps(derivs.tolist()),
            json.dumps(fnn_rates.tolist())
        ])

        pyplot.subplot(len(derivs_tup), 2, 2*i + 1)
        pyplot.plot(numpy.arange(derivs.shape[0]), derivs)
        pyplot.title('starting with {0}'.format(delays[:(i+1)]))
        pyplot.xlabel('next delay')
        pyplot.ylabel('geo. mean deriv. toward NN')

        pyplot.subplot(len(derivs_tup), 2, 2*i + 2)
        pyplot.plot(numpy.arange(fnn_rates.shape[0]), fnn_rates)
        pyplot.title('starting with {0}'.format(delays[:(i+1)]))
        pyplot.xlabel('next delay')
        pyplot.ylabel('FNN rate')
    pyplot.savefig('nichkawde-{0}-causes-{1}.png'.format(cname, ename))
    pyplot.close(fig)

def run_ccm_bootstraps(cname, ename, embedding, cause, L, theiler_window):
    assert isinstance(embedding, pyembedding.Embedding)

    corrs = []

    for i in range(N_CCM_BOOTSTRAPS):
        sampled_embedding = embedding.sample_embedding(L, replace=True, rng=rng)
        ccm_result, y_actual, y_pred = sampled_embedding.ccm(embedding, cause, theiler_window=theiler_window)

        corrs.append(ccm_result.correlation)

    corrs = numpy.array(corrs)

    db.execute('CREATE TABLE IF NOT EXISTS ccm_correlation_dist (cause, effect, L, delays, mean, sd, pvalue_positive, q0, q1, q2_5, q5, q25, q50, q75, q95, q97_5, q99, q100)')
    db.execute(
        'INSERT INTO ccm_correlation_dist VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        [cname, ename, L, str(embedding.delays), corrs.mean(), corrs.std(), statutils.inverse_quantile(corrs, 0.0).tolist()] +
            numpy.percentile(corrs, [0, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 100]).tolist()
    )
    
    db.execute('CREATE TABLE IF NOT EXISTS ccm_correlations (cause, effect, L, delays, correlation)')
    for corr in corrs:
        db.execute(
            'INSERT INTO ccm_correlations VALUES (?,?,?,?,?)',
            [cname, ename, L, str(embedding.delays), corr]
    )

    return numpy.array(corrs)

main()
