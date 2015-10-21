#!/usr/bin/env python
'''Example simulation + CCM analysis script. Should be copied and modified for particular experiments.
Can be used by itself
'''

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
import sqlite3
import numpy
import matplotlib
import random
matplotlib.use('Agg')
from matplotlib import pyplot

# Needs to point to the right place for pyembedding
sys.path.append(os.path.join(SCRIPT_DIR, 'pyembedding'))
import pyembedding
import uzalcost
import jsonobject
import models
import statutils

# Various parameters
save_correlations = True
n_ccm_bootstraps = 1000

# Connect to output database
if os.path.exists('results.sqlite'):
    sys.stderr.write('Output database present. Aborting.\n')
    sys.exit(1)
db = sqlite3.connect('results.sqlite')

# Load job info and record in database
if os.path.exists('job_info.json'):
    job_info = jsonobject.load_from_file('job_info.json')
else:
    job_info = jsonobject.JSONObject()
    job_info.job_id = 0
    job_info.random_seed = random.SystemRandom().randint(1, 2**31-1)
db.execute('CREATE TABLE job_info ({0})'.format(', '.join([key for key in job_info.keys()])))
db.execute('INSERT INTO job_info VALUES ({0})'.format(', '.join(['?'] * len(job_info))), job_info.values())

# Set up RNG
rng = numpy.random.RandomState(job_info.random_seed)

# Initialize SIR model parameters: first defaults,
# then load overrides from JSON file if present
params = jsonobject.JSONObject(
    random_seed = rng.randint(1, 2**31 - 1),
    n_pathogens = 2,

    dt_euler = 0.01,
    adaptive = False,
    tol = 1e-6,
    t_end = 72000.0,
    dt_output = 30.0,

    mu = 0.00054794520548,
    nu = [0.2, 0.2],
    gamma = [0.0, 0.0],

    beta0 = [0.22, 0.22],
    S_init = [1.0, 1.0], # Initialized below
    I_init = [0.0, 0.0], # Initialized below
    beta_change_start = [0.0, 0.0],
    beta_slope = [0.0, 0.0],
    psi = [360.0, 360.0],
    omega = [0.0, 0.0],
    eps = [0.1, 0.1],
    sigma = [[1.0, 0.01], [0.0, 1.0]],

    shared_proc = False,
    sd_proc = [0.00, 0.00],

    shared_obs = False,
    sd_obs = [0.0, 0.0],

    shared_obs_C = False,
    sd_obs_C = [1e-2, 1e-2]
)
params.S_init = rng.uniform(0.0, 1.0, size=params.n_pathogens)
params.I_init = rng.uniform(0.0, 1.0 - params.S_init)
if os.path.exists('sir_params.json'):
    params.update_from_file('sir_params.json')

# main(): gets called at the end (after other functions have been defined)
def main():
    # Run a real simulation
    sir_out = run_simulation()
    C0 = sir_out.C[-1200:, 0]
    C1 = sir_out.C[-1200:, 1]

    # Or, comment out the above and uncomment the following to run a fake simulation:
    # C0 = rng.normal(0, 1, size=1200)
    # C1 = rng.normal(0, 1, size=1200)

    plot_timeseries('C0', C0)
    plot_timeseries('C1', C1)

    run_analysis('C0', C0, 'C1', C1)
    run_analysis('C1', C1, 'C0', C0)

    db.commit()
    db.close()

def run_simulation():
    try:
        db.execute('CREATE TABLE sir_params (job_id INTEGER, params TEXT)')
        db.execute('INSERT INTO sir_params VALUES (?,?)', (job_info.job_id, params.dump_to_string()))

        return models.run_via_pypy('multistrain_sde', params)
    except models.ExecutionException as e:
        sys.stderr.write('An exception occurred trying to run simulation...\n')
        sys.stderr.write('{0}\n'.format(e.cause))
        sys.stderr.write(e.stderr_data)
        sys.exit(1)

def plot_timeseries(name, timeseries):
    fig = pyplot.figure()
    pyplot.plot(timeseries)
    pyplot.xlabel('time (months)')
    pyplot.ylabel(name)
    pyplot.savefig(name + '.png')
    pyplot.close(fig)

def run_analysis(cname, cause, ename, effect):
    '''Run analysis for a single causal direction.'''
    sys.stderr.write('Running {0}-causes-{1}\n'.format(cname, ename))

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
    assert max_window > prediction_horizon

    # Run Uzal cost function (will implicitly compile Uzal's C code if necessary)
    ms, Lks, params = uzalcost.run_uzal_costfunc(
        effect, theiler_window=theiler_window, max_prediction_horizon=prediction_horizon,
        max_window=max_window
    )
    best_m_index = numpy.argmin(Lks)
    embedding_dimension = ms[best_m_index]
    sys.stderr.write('  Uzal full embedding dimension = {0} (Lk = {1})\n'.format(embedding_dimension, Lks[best_m_index]))

    # Run Nichkawde algorithm to identify sub-embedding
    full_embedding = pyembedding.Embedding(effect, range(embedding_dimension))
    sub_embedding = full_embedding.nichkawde_subembedding(theiler_window)
    sys.stderr.write('  Nichkawde sub-embedding: {0}\n'.format(sub_embedding.delays))

    # min library size: embedding_dimension + 2,
    # so vectors should usually have embedding_dimension + 1 neighbors available
    Lmin = sub_embedding.embedding_dimension + 2

    # max library size: just the number of available delay vectors
    Lmax = sub_embedding.delay_vector_count

    sys.stderr.write('  Using Lmin = {0}, Lmax = {1}\n'.format(Lmin, Lmax))

    assert Lmax > Lmin

    corrs_Lmin = run_ccm_bootstraps(cname, ename, sub_embedding, cause, Lmin, theiler_window)
    corrs_Lmax = run_ccm_bootstraps(cname, ename, sub_embedding, cause, Lmax, theiler_window)

    db.execute(
        'CREATE TABLE IF NOT EXISTS ccm_increase (job_id, cause, effect, Lmin, Lmax, delays, pvalue_increase)'
    )
    db.execute(
        'INSERT INTO ccm_increase VALUES (?,?,?,?,?,?,?)',
        [job_info.job_id, cname, ename, Lmin, Lmax, str(sub_embedding.delays), 1.0 - numpy.mean(statutils.inverse_quantile(corrs_Lmin, corrs_Lmax))]
    )

def run_ccm_bootstraps(cname, ename, embedding, cause, L, theiler_window):
    assert isinstance(embedding, pyembedding.Embedding)

    corrs = []

    for i in range(n_ccm_bootstraps):
        sampled_embedding = embedding.sample_embedding(L, replace=True, rng=rng)
        ccm_result, y_actual, y_pred = sampled_embedding.ccm(embedding, cause, theiler_window=theiler_window)

        corrs.append(ccm_result.correlation)

    corrs = numpy.array(corrs)

    db.execute('CREATE TABLE IF NOT EXISTS ccm_correlation_dist (job_id, cause, effect, L, delays, mean, sd, pvalue_positive, q0, q1, q2_5, q5, q25, q50, q75, q95, q97_5, q99, q100)')
    db.execute(
        'INSERT INTO ccm_correlation_dist VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        [job_info.job_id, cname, ename, L, str(embedding.delays), corrs.mean(), corrs.std(), statutils.inverse_quantile(corrs, 0.0).tolist()] +
            numpy.percentile(corrs, [0, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 100]).tolist()
    )

    if save_correlations:
        db.execute('CREATE TABLE IF NOT EXISTS ccm_correlations (job_id, cause, effect, L, delays, correlation)')
        for corr in corrs:
            db.execute(
                'INSERT INTO ccm_correlations VALUES (?,?,?,?,?,?)',
                [job_info.job_id, cname, ename, L, str(embedding.delays), corr]
        )

    return numpy.array(corrs)

main()
