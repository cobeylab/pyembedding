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
import jsonobject
import models
import statutils
import npybuffer

# Make sure this is an absolute path
#SIM_DB_PATH = '/Users/ebaskerv/uchicago/midway_cobey/2015-10-23-simulations/results_gathered.sqlite'
SIM_DB_PATH = '/project/cobey/ccmproject-storage/2015-10-23-simulations/results_gathered.sqlite'

SIMULATION_SAMPLES_PER_YEAR = 36    # This is an assumption about the time series being loaded; don't change
                                    # unless a new simulation DB is being used with different settings.
                                    # 36 implies 10-day samples in a 360-day year in the original simulations.

TS_X_AXIS = 'time (years)'          # Labels for time-series plot. Rewrite to match settings
TS_Y_AXIS = 'monthly cases'         # below.

CCM_YEARS = 100                     # Number of years to use at end of time series

CCM_SAMPLES_PER_YEAR = 12           # E.g., 12 implies 30-day samples in a 360-day year.
                                    # Must divide evenly into SIMULATION_SAMPLES_PER_YEAR

MAX_THEILER_WINDOW = 60             # Maximum window for exclude nearest neighbors temporally

MAX_PREDICTION_HORIZON = 120        # Twice the Theiler window is used to set the prediction
                                    # horizon for the Uzal cost function; it is also bounded
                                    # by this value.

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

STANDARDIZE = False                 # If True, each time series is standardized to mean=0, sd=1.
                                    # This is applied after all other transformations.

# Runs Uzal cost function to find upper bound on Nichkawde embedding lags, and then does bootstrapped CCM
# at Lmin, Lmax for that embedding
EMBEDDING_ALGORITHM = 'uzal_nichkawde'
UZAL_FACTOR = 2.0                   # Multiplies Uzal upper bound by this much
OVERRIDE_UZAL_UPPER_BOUND = None    # If not None, skip Uzal algorithm and use this Nichkawde bound instead

# Runs all valid E/tau combinations: SWEEP_EMBEDDING_DIMENSIONS x SWEEP_DELAYS
# EMBEDDING_ALGORITHM = 'uniform_sweep'

# Searches for E/tau combination with highest CCM rho at Lmax, and then does bootstrapped CCM
# at Lmin, Lmax for chosen E/tau combinations
# EMBEDDING_ALGORITHM = 'max_ccm_rho'

# Searches for E/tau combination with highest univariate prediction for effect variable, and then does bootstrapped CCM
# at Lmin, Lmax for chosen E/tau combinations
# EMBEDDING_ALGORITHM = 'max_univariate_prediction'

# These lists control the uniform_sweep, max_ccm_rho, and max_univariate_prediction modes above
SWEEP_EMBEDDING_DIMENSIONS = range(1, 11)
SWEEP_DELAYS = [1, 2, 4]

N_CCM_BOOTSTRAPS = 1000

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
    
    x0name = VARIABLE_NAME + '0'
    x1name = VARIABLE_NAME + '1'
    
    plot_timeseries([x0, x1], [x0name, x1name], TS_X_AXIS, TS_Y_AXIS, 'timeseries.png')
    
    run_analysis(x0name, x0, x1name, x1)
    run_analysis(x1name, x1, x0name, x0)

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
    
    if STANDARDIZE:
        for i in range(arr_mod.shape[1]):
            arr_mod[:,i] -= numpy.mean(arr_mod[:,i])
            arr_mod[:,i] /= numpy.std(arr_mod[:,i])
    
    if FIRST_DIFFERENCE:
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
    if cause_sd == 0.0 or effect_sd == 0.0:
        sys.stderr.write('No variation cause or effect time series; skipping analysis.\n')
        return
    else:
        # Identify delay at which autocorrelation drops to 1/e
        ac_delay, autocorr = pyembedding.autocorrelation_threshold_delay(effect, 1.0/numpy.e)
        sys.stderr.write('  ac_delay, autocorr = {0}, {1}\n'.format(ac_delay, autocorr))
        
        # Calculate Theiler window (limit on closeness of neighbors in time)
        theiler_window = min(MAX_THEILER_WINDOW, 3 * ac_delay)
        sys.stderr.write('  theiler_window = {0}\n'.format(theiler_window))
        assert theiler_window < effect.shape[0]
        
        if EMBEDDING_ALGORITHM == 'uzal_nichkawde':
            run_analysis_uzal_nichkawde(cname, cause, ename, effect, theiler_window)
        elif EMBEDDING_ALGORITHM == 'uniform_sweep':
            run_analysis_uniform_sweep(cname, cause, ename, effect, theiler_window)
        elif EMBEDDING_ALGORITHM == 'max_ccm_rho':
            run_analysis_max_ccm_rho(cname, cause, ename, effect, theiler_window)
        elif EMBEDDING_ALGORITHM == 'max_univariate_prediction':
            run_analysis_max_univariate_prediction(cname, cause, ename, effect, theiler_window)

def run_analysis_uzal_nichkawde(cname, cause, ename, effect, theiler_window):
    # Calculate maximum prediction horizon (used by Uzal cost function)
    prediction_horizon = min(MAX_PREDICTION_HORIZON, 2 * theiler_window)
    sys.stderr.write('  prediction_horizon = {0}\n'.format(prediction_horizon))
    assert prediction_horizon > theiler_window

    # Hard-code maximum delay vector window:, equal to delay * (embedding_dimension - 1)
    max_window = 60
    sys.stderr.write('  max_window = {0}\n'.format(max_window))

    # Run Uzal cost function (will implicitly compile Uzal's C code if necessary)
    if OVERRIDE_UZAL_UPPER_BOUND is not None:
        max_embedding_dimension = OVERRIDE_UZAL_UPPER_BOUND
    else:
        ms, Lks, params = uzalcost.run_uzal_costfunc(
            effect, theiler_window=theiler_window, max_prediction_horizon=prediction_horizon,
            max_window=max_window
        )
        best_m_index = numpy.argmin(Lks)
        sys.stderr.write('  Uzal full embedding dimension = {0} (Lk = {1})\n'.format(ms[best_m_index], Lks[best_m_index]))
        max_embedding_dimension = int(numpy.round(UZAL_FACTOR * ms[best_m_index]))
    sys.stderr.write('  Using max embedding dimension = {}\n'.format(max_embedding_dimension))

    # Run Nichkawde algorithm to identify sub-embedding
    embedding, derivs_tup, fnn_rates_tup = pyembedding.nichkawde_embedding(effect, theiler_window, max_embedding_dimension, return_metrics=True)
    delays = embedding.delays
    write_and_plot_nichkawde_metrics(cname, ename, delays, derivs_tup, fnn_rates_tup)

    sys.stderr.write('  Nichkawde sub-embedding: {0}\n'.format(delays))
    run_analysis_for_embedding(cname, cause, ename, effect, embedding, theiler_window)

def run_analysis_uniform_sweep(cname, cause, ename, effect, theiler_window):
    for E in SWEEP_EMBEDDING_DIMENSIONS:
        for tau in (SWEEP_DELAYS if E > 1 else [1]):
            sys.stderr.write('  Running for E={}, tau={}\n'.format(E, tau))
            delays = tuple(range(0, E*tau, tau))
            embedding = pyembedding.Embedding(effect, delays=delays)
            if embedding.delay_vector_count < embedding.embedding_dimension + 2:
                sys.stderr.write('  Lmax < Lmin; skipping E={}, tau={}\n'.format(E, tau))
                continue

            run_analysis_for_embedding(cname, cause, ename, effect, embedding, theiler_window)

def run_analysis_max_ccm_rho(cname, cause, ename, effect, theiler_window):
    max_corr = float('-inf')
    max_corr_emb = None
    max_corr_Etau = None
    for E in SWEEP_EMBEDDING_DIMENSIONS:
        for tau in (SWEEP_DELAYS if E > 1 else [1]):
            delays = tuple(range(0, E*tau, tau))
            embedding = pyembedding.Embedding(effect, delays=delays)
            if embedding.delay_vector_count < embedding.embedding_dimension + 2:
                sys.stderr.write('  Lmax < Lmin; skipping E={}, tau={}\n'.format(E, tau))
                continue

            corr = run_ccm(cname, ename, embedding, cause, theiler_window)
            sys.stderr.write('  corr for E={}, tau={} : {}\n'.format(E, tau, corr))
            if corr > max_corr:
                max_corr = corr
                max_corr_emb = embedding
                max_corr_Etau = (E, tau)
    sys.stderr.write('  Using E={}, tau = {}\n'.format(*max_corr_Etau))
    run_analysis_for_embedding(cname, cause, ename, effect, max_corr_emb, theiler_window)

def run_analysis_max_univariate_prediction(cname, cause, ename, effect, theiler_window):
    max_corr = float('-inf')
    max_corr_Etau = None
    for E in SWEEP_EMBEDDING_DIMENSIONS:
        for tau in SWEEP_DELAYS:
            delays = tuple(range(0, E*tau, tau))
            embedding = pyembedding.Embedding(effect[:-1], delays)
            if embedding.delay_vector_count < embedding.embedding_dimension + 2:
                sys.stderr.write('  Lmax < Lmin; skipping E={}, tau={}\n'.format(E, tau))
                continue

            eff_off, eff_off_pred = embedding.simplex_predict_using_embedding(embedding, effect[1:], theiler_window=theiler_window)
            corr = numpy.corrcoef(eff_off, eff_off_pred)[0,1]

            db.execute('CREATE TABLE IF NOT EXISTS univariate_predictions (variable, delays, correlation)')
            db.execute('INSERT INTO univariate_predictions VALUES (?,?,?)', [ename, str(delays), corr])

            sys.stderr.write('  corr for E={}, tau={} : {}\n'.format(E, tau, corr))
            if corr > max_corr:
                max_corr = corr
                max_corr_Etau = (E, tau)
    E, tau = max_corr_Etau
    delays = tuple(range(0, E*tau, tau))
    max_corr_emb = pyembedding.Embedding(effect, delays)
    sys.stderr.write('  Using E = {}, tau = {}\n'.format(*max_corr_Etau))
    run_analysis_for_embedding(cname, cause, ename, effect, max_corr_emb, theiler_window)

def run_analysis_for_embedding(cname, cause, ename, effect, embedding, theiler_window):
    # min library size: embedding_dimension + 2,
    # so vectors should usually have embedding_dimension + 1 neighbors available
    Lmin = embedding.embedding_dimension + 2

    # max library size: just the number of available delay vectors
    Lmax = embedding.delay_vector_count

    assert Lmax > Lmin
    sys.stderr.write('  Using Lmin = {}, Lmax = {}\n'.format(Lmin, Lmax))

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

def run_ccm(cname, ename, embedding, cause, theiler_window):
    assert isinstance(embedding, pyembedding.Embedding)

    ccm_result, y_actual, y_pred = embedding.ccm(embedding, cause, theiler_window=theiler_window)
    db.execute('CREATE TABLE IF NOT EXISTS ccm_correlations_single (cause, effect, L, delays, correlation)')
    db.execute(
        'INSERT INTO ccm_correlations_single VALUES (?,?,?,?,?)',
        [cname, ename, embedding.delay_vector_count, str(embedding.delays), ccm_result.correlation]
    )

    return ccm_result.correlation

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


def plot_timeseries(series, labels, xlabel, ylabel, filename):
    fig = pyplot.figure(figsize=(12,5))
    for x in series:
        pyplot.plot(x)
    pyplot.legend(labels)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.savefig(filename)
    pyplot.close(fig)

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

main()
