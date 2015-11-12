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
    job_info = runmany_info['job_info']

    # Make sure simulation database is present
    if not os.path.exists(sim_db_path):
        sys.stderr.write('Simulation database not present; aborting\n')
        sys.exit(1)

    # Connect to output database
    if os.path.exists('results.sqlite'):
        sys.stderr.write('Output database present. Aborting.\n')
        sys.exit(1)
    db = sqlite3.connect('results.sqlite')
    job_id = runmany_info['simulation_job_id']
    db.execute('CREATE TABLE job_info ({0})'.format(', '.join([key for key in job_info.keys()])))
    db.execute('INSERT INTO job_info VALUES ({0})'.format(', '.join(['?'] * len(job_info))), job_info.values())

    ccm_settings = runmany_info['ccm_settings']

    # Set up RNG
    rng = numpy.random.RandomState(job_info['random_seed'])
    X = load_simulation(sim_db_path, job_id, ccm_settings)
    if numpy.any(numpy.logical_or(
        numpy.isnan(X),
        numpy.isinf(X)
    )):
        sys.stderr.write('nans or infs in data; skipping all analyses.\n')
        sys.exit(0)
    
    x0 = X[:,0]
    x1 = X[:,1]

    variable_name = ccm_settings['variable_name']
    x0name = variable_name + '0'
    x1name = variable_name + '1'
    
    plot_timeseries([x0, x1], [x0name, x1name], ccm_settings['timeseries_x_label'], ccm_settings['timeseries_y_label'], 'timeseries.png')
    
    run_analysis(x0name, x0, x1name, x1, db, rng, ccm_settings)
    run_analysis(x1name, x1, x0name, x0, db, rng, ccm_settings)

    db.commit()
    db.close()

def load_simulation(sim_db_path, job_id, ccm_settings):
    '''Loads and processes time series based on settings at top of file.'''
    with sqlite3.connect(sim_db_path) as sim_db:
        buf = sim_db.execute(
            'SELECT {} FROM timeseries WHERE job_id = ?'.format(ccm_settings['variable_name']),
            [job_id]
        ).next()[0]
        assert isinstance(buf, buffer)
        arr = npybuffer.npy_buffer_to_ndarray(buf)
    assert arr.shape[1] == 2

    years = ccm_settings['years']
    simulation_samples_per_year = ccm_settings['simulation_samples_per_year']
    ccm_samples_per_year = ccm_settings['ccm_samples_per_year']

    # Get the unthinned sample from the end of the time series
    sim_samps_unthinned = years * simulation_samples_per_year
    thin = simulation_samples_per_year / ccm_samples_per_year
    arr_end_unthinned = arr[-sim_samps_unthinned:, :]
    
    # Thin the samples, adding in the intervening samples if requested
    arr_mod = arr_end_unthinned[::thin, :]
    if ccm_settings['add_samples']:
        for i in range(1, thin):
            arr_mod += arr_end_unthinned[i::thin, :]
    
    if ccm_settings['log_transform']:
        arr_mod = numpy.log(arr_mod)
    
    if ccm_settings['first_difference']:
        arr_mod = arr_mod[1:, :] - arr_mod[:-1, :]
    
    if ccm_settings['standardize']:
        for i in range(arr_mod.shape[1]):
            arr_mod[:,i] -= numpy.mean(arr_mod[:,i])
            arr_mod[:,i] /= numpy.std(arr_mod[:,i])


    if ccm_settings['first_difference']:
        assert arr_mod.shape[0] == years * ccm_samples_per_year - 1
    else:
        assert arr_mod.shape[0] == years * ccm_samples_per_year
    assert arr_mod.shape[1] == 2
    
    return arr_mod

def run_analysis(cname, cause, ename, effect, db, rng, ccm_settings):
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
        theiler_window = min(ccm_settings['max_theiler_window'], 3 * ac_delay)
        sys.stderr.write('  theiler_window = {0}\n'.format(theiler_window))
        assert theiler_window < effect.shape[0]

        embedding_algorithm = ccm_settings['embedding_algorithm']
        if embedding_algorithm == 'uzal_nichkawde':
            run_analysis_uzal_nichkawde(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings)
        elif embedding_algorithm == 'uniform_sweep':
            run_analysis_uniform_sweep(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings)
        elif embedding_algorithm == 'max_ccm_rho':
            run_analysis_max_ccm_rho(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings)
        elif embedding_algorithm == 'max_univariate_prediction':
            run_analysis_max_univariate_prediction(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings)

def run_analysis_uzal_nichkawde(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings):
    # Calculate maximum prediction horizon (used by Uzal cost function)
    prediction_horizon = min(ccm_settings['max_prediction_horizon'], 2 * theiler_window)
    sys.stderr.write('  prediction_horizon = {0}\n'.format(prediction_horizon))
    assert prediction_horizon > theiler_window

    # Hard-code maximum delay vector window:, equal to delay * (embedding_dimension - 1)
    max_window = 60
    sys.stderr.write('  max_window = {0}\n'.format(max_window))

    # Run Uzal cost function (will implicitly compile Uzal's C code if necessary)
    if 'override_uzal_upper_bound' in ccm_settings and ccm_settings['override_uzal_upper_bound'] is not None:
        max_embedding_dimension = ccm_settings['override_uzal_upper_bound']
    else:
        ms, Lks, params = uzalcost.run_uzal_costfunc(
            effect, theiler_window=theiler_window, max_prediction_horizon=prediction_horizon,
            max_window=max_window
        )
        best_m_index = numpy.argmin(Lks)
        sys.stderr.write('  Uzal full embedding dimension = {0} (Lk = {1})\n'.format(ms[best_m_index], Lks[best_m_index]))
        max_embedding_dimension = int(numpy.round(ccm_settings['uzal_factor'] * ms[best_m_index]))
    sys.stderr.write('  Using max embedding dimension = {}\n'.format(max_embedding_dimension))

    # Run Nichkawde algorithm to identify sub-embedding
    embedding, derivs_tup, fnn_rates_tup = pyembedding.nichkawde_embedding(effect, theiler_window, max_embedding_dimension, return_metrics=True)
    delays = embedding.delays
    write_and_plot_nichkawde_metrics(cname, ename, delays, derivs_tup, fnn_rates_tup, db)

    sys.stderr.write('  Nichkawde sub-embedding: {0}\n'.format(delays))
    run_analysis_for_embedding(cname, cause, ename, effect, embedding, theiler_window, ccm_settings['n_ccm_bootstraps'], db, rng)

def run_analysis_uniform_sweep(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings):
    for E in ccm_settings['sweep_embedding_dimensions']:
        for tau in (ccm_settings['sweep_delays'] if E > 1 else [1]):
            sys.stderr.write('  Running for E={}, tau={}\n'.format(E, tau))
            delays = tuple(range(0, E*tau, tau))
            embedding = pyembedding.Embedding(effect, delays=delays)
            if embedding.delay_vector_count < embedding.embedding_dimension + 2:
                sys.stderr.write('  Lmax < Lmin; skipping E={}, tau={}\n'.format(E, tau))
                continue

            run_analysis_for_embedding(cname, cause, ename, effect, embedding, theiler_window, ccm_settings['n_ccm_bootstraps'], db, rng)

def run_analysis_max_ccm_rho(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings):
    max_corr = float('-inf')
    max_corr_emb = None
    max_corr_Etau = None
    for E in ccm_settings['sweep_embedding_dimensions']:
        for tau in (ccm_settings['sweep_delays'] if E > 1 else [1]):
            delays = tuple(range(0, E*tau, tau))
            embedding = pyembedding.Embedding(effect, delays=delays)
            if embedding.delay_vector_count < embedding.embedding_dimension + 2:
                sys.stderr.write('  Lmax < Lmin; skipping E={}, tau={}\n'.format(E, tau))
                continue

            corr = run_ccm(cname, ename, embedding, cause, theiler_window, db)
            sys.stderr.write('  corr for E={}, tau={} : {}\n'.format(E, tau, corr))
            if corr > max_corr:
                max_corr = corr
                max_corr_emb = embedding
                max_corr_Etau = (E, tau)
    sys.stderr.write('  Using E={}, tau = {}\n'.format(*max_corr_Etau))
    run_analysis_for_embedding(cname, cause, ename, effect, max_corr_emb, theiler_window, ccm_settings['n_ccm_bootstraps'], db, rng)

def run_analysis_max_univariate_prediction(cname, cause, ename, effect, theiler_window, db, rng, ccm_settings):
    if 'delta_tau_termination' in ccm_settings:
        delta_tau_termination = ccm_settings['delta_tau_termination']
    else:
        delta_tau_termination = None
    
    max_corr = float('-inf')
    max_corr_Etau = None
    for E in ccm_settings['sweep_embedding_dimensions']:
        if delta_tau_termination is not None:
            max_corr_this_E = float('-inf')
            max_corr_tau_this_E = None
        
        for tau in (ccm_settings['sweep_delays'] if E > 1 else [1]):
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
            
            if delta_tau_termination is not None:
                if corr > max_corr_this_E:
                    max_corr_this_E = corr
                    max_corr_tau_this_E = tau
            
                if E * (tau - max_corr_tau_this_E) >= delta_tau_termination:
                    sys.stderr.write('{} taus since a maximum for this E; assuming found.\n'.format(delta_tau_termination))
                    break
    
    E, tau = max_corr_Etau
    delays = tuple(range(0, E*tau, tau))
    max_corr_emb = pyembedding.Embedding(effect, delays)
    sys.stderr.write('  Using E = {}, tau = {}\n'.format(*max_corr_Etau))
    run_analysis_for_embedding(cname, cause, ename, effect, max_corr_emb, theiler_window, ccm_settings['n_ccm_bootstraps'], db, rng)

def run_analysis_for_embedding(cname, cause, ename, effect, embedding, theiler_window, n_bootstraps, db, rng):
    # min library size: embedding_dimension + 2,
    # so vectors should usually have embedding_dimension + 1 neighbors available
    Lmin = embedding.embedding_dimension + 2

    # max library size: just the number of available delay vectors
    Lmax = embedding.delay_vector_count

    assert Lmax > Lmin
    sys.stderr.write('  Using Lmin = {}, Lmax = {}\n'.format(Lmin, Lmax))

    corrs_Lmin = run_ccm_bootstraps(cname, ename, embedding, cause, Lmin, theiler_window, n_bootstraps, db, rng)
    corrs_Lmax = run_ccm_bootstraps(cname, ename, embedding, cause, Lmax, theiler_window, n_bootstraps, db, rng)

    db.execute(
        'CREATE TABLE IF NOT EXISTS ccm_increase (cause, effect, Lmin, Lmax, delays, pvalue_increase)'
    )
    db.execute(
        'INSERT INTO ccm_increase VALUES (?,?,?,?,?,?)',
        [cname, ename, Lmin, Lmax, str(embedding.delays), 1.0 - numpy.mean(statutils.inverse_quantile(corrs_Lmin, corrs_Lmax))]
    )
    db.commit()

def run_ccm(cname, ename, embedding, cause, theiler_window, db):
    assert isinstance(embedding, pyembedding.Embedding)

    ccm_result, y_actual, y_pred = embedding.ccm(embedding, cause, theiler_window=theiler_window)
    db.execute('CREATE TABLE IF NOT EXISTS ccm_correlations_single (cause, effect, L, delays, correlation)')
    db.execute(
        'INSERT INTO ccm_correlations_single VALUES (?,?,?,?,?)',
        [cname, ename, embedding.delay_vector_count, str(embedding.delays), ccm_result['correlation']]
    )

    return ccm_result['correlation']

def run_ccm_bootstraps(cname, ename, embedding, cause, L, theiler_window, n_bootstraps, db, rng):
    assert isinstance(embedding, pyembedding.Embedding)

    corrs = []

    for i in range(n_bootstraps):
        sampled_embedding = embedding.sample_embedding(L, replace=True, rng=rng)
        ccm_result, y_actual, y_pred = sampled_embedding.ccm(embedding, cause, theiler_window=theiler_window)

        corrs.append(ccm_result['correlation'])

    corrs = numpy.array(corrs)

    db.execute('CREATE TABLE IF NOT EXISTS ccm_correlation_dist (cause, effect, L, delays, mean, sd, pvalue_positive, q0, q1, q2_5, q5, q25, q50, q75, q95, q97_5, q99, q100)')
    db.execute(
        'INSERT INTO ccm_correlation_dist VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        [cname, ename, L, str(embedding.delays), corrs.mean(), corrs.std(), statutils.inverse_quantile(corrs, 0.0).tolist()] +
            [x for x in numpy.percentile(corrs, [0, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 100])]
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

def write_and_plot_nichkawde_metrics(cname, ename, delays, derivs_tup, fnn_rates_tup, db):
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

def load_json(filename):
    with open(filename) as f:
        return json.load(f, object_pairs_hook=OrderedDict)

main()
