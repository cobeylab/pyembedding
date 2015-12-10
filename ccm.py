#!/usr/bin/env python

import argparse

import os
import sys
import sqlite3
import json
import pyembedding
import statutils
import multiprocessing
import numpy
from collections import OrderedDict
import csv
import random

def main():
    args = parse_arguments()
    
    E = args.embedding_dimension
    Emin = args.min_embedding_dimension
    Emax = args.max_embedding_dimension
    
    tau = args.embedding_lag
    taumin = args.min_embedding_lag
    taumax = args.max_embedding_lag
    
    dt = args.temporal_separation
    
    max_lag = args.max_cross_map_lag
    n_bootstraps = args.bootstraps
    
    cores = args.cores
    
    # Check input & output files
    if args.input_filename == args.output_filename:
        parser.exit('Input filename and output filename cannot be the same.')
    if os.path.exists(args.output_filename):
        if args.overwrite_output:
            os.remove(args.output_filename)
        else:
            sys.stderr.write('Output filename {0} exists. Delete it or use --overwrite-output.\n'.format(args.output_filename))
            sys.exit(1)
    
    # Load input data
    data = load_data(args.input_filename, args.variable, args.table)
    
    # Set up output data
    db = sqlite3.connect(args.output_filename)
    
    # Identify E, tau for each variable using univariate one-step-ahead prediction
    if E is None:
        Etau_dict = identify_Etau(db, data, Emin, Emax, taumin, taumax, dt, cores)
    else:
        Etau_dict = OrderedDict([(var_name, (E, tau)) for var_name in data.keys()])
    
    for cname, cause in data.iteritems():
        for ename, effect in data.iteritems():
            if cname != ename:
                E, tau = Etau_dict[ename]
                if E is None:
                    sys.stderr.write('Skipping {} as effect\n'.format(ename))
                    continue
                
                sys.stderr.write('Running {}, {}\n'.format(cname, ename))
                run_analysis(db, cname, cause, ename, effect, E, tau, dt, max_lag, n_bootstraps, cores)

def identify_Etau(db, data, Emin, Emax, taumin, taumax, dt, cores):
    assert Emin > 0
    assert taumin > 0
    
    Etau_list = []
    for E in range(Emin, Emax + 1):
        if E == 1:
            Etau_list.append((1, taumin))
        else:
            for tau in range(taumin, taumax + 1):
                Etau_list.append((E, tau))
    
    Etau_dict = OrderedDict()
    
    db.execute('CREATE TABLE IF NOT EXISTS Etau (variable, E, tau)')
    for var_name in data.keys():
        E, tau = identify_Etau_single(data[var_name], Etau_list, dt, cores)
        print var_name, E, tau
        Etau_dict[var_name] = (E, tau)
        db.execute('INSERT INTO Etau VALUES (?,?,?)', [var_name, E, tau])
    db.commit()
    
    return Etau_dict
    
def identify_Etau_single(x, Etau_list, dt, cores):
    E, tau, corrs = pyembedding.identify_embedding_max_univariate_prediction(x, Etau_list, dt, cores)
    
    return E, tau

def run_analysis(db, cname, cause, ename, effect, E, tau, dt, max_lag, n_bootstraps, cores):
#     
#     L = args.library_size
#     assert L is None
    
    pool = multiprocessing.Pool(cores)
    
    args = [(cause, effect, E, tau, dt, lag, n_bootstraps) for lag in range(-max_lag, max_lag + 1)]
    
    async_result = pool.map_async(run_analysis_mappable, args, chunksize=1)
    try:
        # Need a timeout to avoid KeyboardInterrupt Python bug; a million years should be safe
        results = async_result.get(60*60*24*365*1000*1000)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.stdout.write('\n')
    
    db.execute('CREATE TABLE IF NOT EXISTS correlations (cause, effect, lag, L, correlation)')
    db.execute('CREATE TABLE IF NOT EXISTS tests (cause, effect, lag, Lmin, Lmax, pval_positive, pval_increase)')
    db.execute('CREATE TABLE IF NOT EXISTS lagtests (cause, effect, best_lag, pval_positive, pval_increase, pval_neg_best, pval_nonpos_best)')
    
    best_lag_neg = None
    best_lag_neg_corr_med = None
    best_lag_neg_corrs = None
    best_lag_neg_pval_increase = None
    best_lag_neg_pval_positive = None
    
    best_lag_pos = None
    best_lag_pos_corr_med = None
    best_lag_pos_corrs = None
    best_lag_pos_pval_increase = None
    best_lag_pos_pval_positive = None
    
    zero_corrs = None
    zero_corr_med = None
    zero_pval_increase = None
    zero_pval_positive = None
    
    for lag, results_dict in results:
        Ls = results_dict.keys()
        Lmin = min(Ls)
        Lmax = max(Ls)
        
        corrs_Lmin = results_dict[Lmin]
        corrs_Lmax = results_dict[Lmax]
        
        pval_positive = statutils.inverse_quantile(corrs_Lmax, 0.0).tolist()
        pval_increase = 1.0 - numpy.mean(statutils.inverse_quantile(corrs_Lmin, corrs_Lmax))
        
        db.execute(
            'INSERT INTO tests VALUES (?,?,?,?,?,?,?)',
            [cname, ename, lag, Lmin, Lmax, pval_positive, pval_increase]
        )
        
        corrs = corrs_Lmax
        corr_med = numpy.median(corrs)
        
        if lag == 0:
            zero_corr_med = corr_med
            zero_corrs = corrs
            zero_pval_increase = pval_increase
            zero_pval_positive = pval_positive
        elif lag < 0 and (best_lag_neg is None or corr_med > best_lag_neg_corr_med):
            best_lag_neg = lag
            best_lag_neg_corr_med = corr_med
            best_lag_neg_corrs = corrs
            best_lag_neg_pval_increase = pval_increase
            best_lag_neg_pval_positive = pval_positive
        elif lag > 0 and (best_lag_pos is None or corr_med > best_lag_pos_corr_med):
            best_lag_pos = lag
            best_lag_pos_corr_med = corr_med
            best_lag_pos_corrs = corrs
            best_lag_pos_pval_increase = pval_increase
            best_lag_pos_pval_positive = pval_positive
        
        for L in Ls:
            corrs = results_dict[L]
            for corr in corrs:
                db.execute(
                    'INSERT INTO correlations VALUES (?,?,?,?,?)',
                    [cname, ename, lag, L, corr]
                )
    
    # Get the best negative-or-zero lag
    if best_lag_neg_corr_med > zero_corr_med:
        best_lag_nonpos = best_lag_neg
        best_lag_nonpos_corrs = best_lag_neg_corrs
        best_lag_nonpos_corr_med = best_lag_neg_corr_med
        best_lag_nonpos_pval_increase = best_lag_neg_pval_increase
        best_lag_nonpos_pval_positive = best_lag_neg_pval_positive
    else:
        best_lag_nonpos = 0
        best_lag_nonpos_corrs = zero_corrs
        best_lag_nonpos_corr_med = zero_corr_med
        best_lag_nonpos_pval_increase = zero_pval_increase
        best_lag_nonpos_pval_positive = zero_pval_positive
    
    # Get the best positive-or-zero lag
    if best_lag_pos_corr_med > zero_corr_med:
        best_lag_nonneg = best_lag_pos
        best_lag_nonneg_corrs = best_lag_pos_corrs
        best_lag_nonneg_corr_med = best_lag_pos_corr_med
        best_lag_nonneg_pval_increase = best_lag_pos_pval_increase
        best_lag_nonneg_pval_positive = best_lag_pos_pval_positive
    else:
        best_lag_nonneg = 0
        best_lag_nonneg_corrs = zero_corrs
        best_lag_nonneg_corr_med = zero_corr_med
        best_lag_nonneg_pval_increase = zero_pval_increase
        best_lag_nonneg_pval_positive = zero_pval_positive
    
    # Test if negative is better than nonnegative
    pval_neg_best = 1.0 - numpy.mean(statutils.inverse_quantile(best_lag_nonneg_corrs, best_lag_neg_corrs))
    
    # Test if nonpositive is better than positive
    pval_nonpos_best = 1.0 - numpy.mean(statutils.inverse_quantile(best_lag_pos_corrs, best_lag_nonpos_corrs))
    
    if best_lag_neg_corr_med > best_lag_pos_corr_med and best_lag_neg_corr_med > zero_corr_med:
        best_lag = best_lag_neg
        pval_increase = best_lag_neg_pval_increase
        pval_positive = best_lag_neg_pval_positive
    elif best_lag_pos_corr_med > best_lag_neg_corr_med and best_lag_pos_corr_med > zero_corr_med:
        best_lag = best_lag_pos
        pval_increase = best_lag_pos_pval_increase
        pval_positive = best_lag_pos_pval_positive
    else:
        best_lag = 0
        pval_increase = zero_pval_increase
        pval_positive = zero_pval_positive
    
    db.execute(
        'INSERT INTO lagtests VALUES (?,?,?,?,?,?,?)',
        [cname, ename, best_lag, pval_positive, pval_increase, pval_neg_best, pval_nonpos_best]
    )
    
    db.commit()
            

def run_analysis_mappable(args):
    rng_seed = random.SystemRandom().randint(1, 2**31 - 1)
    rng = numpy.random.RandomState(rng_seed)
    
    cause, effect, E, tau, dt, lag, n_bootstraps = args
    delays = range(lag, lag + E*tau, tau)
    
    emb = pyembedding.Embedding(effect, delays)
    
    Lmin = emb.embedding_dimension + 2
    Lmax = emb.delay_vector_count
    
    results_dict = OrderedDict()
    for L in (Lmin, Lmax):
        corrs = []
        for i in range(n_bootstraps):
            emb_samp = emb.sample_embedding(L, match_valid_vec=cause, replace=True, rng=rng)
            ccm_result, y_actual, y_pred = emb_samp.simplex_predict_summary(emb, cause, theiler_window=dt)

            corrs.append(ccm_result['correlation'])
        
        results_dict[L] = corrs
    
    return lag, results_dict
    

def parse_arguments():
    # Construct arguments
    parser = argparse.ArgumentParser(
        description='Run a convergent cross-mapping (CCM) analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_filename', metavar='<input-file>', type=str, help='Input filename (.sqlite or .csv format).'
    )
    parser.add_argument(
        '--table', metavar='<input-table-name>', type=str, help='Name of data table in SQLite input.'
    )
    parser.add_argument(
        'output_filename', metavar='<output-file>', type=str,
        help='Output filename (SQLite format).'
    )
    parser.add_argument(
        '--variable', '-V', '-v', metavar='<variable-name>', action='append',
        help='Variable (column name in input file) to include. If none are specified, all columns are used.'
    )
    parser.add_argument(
        '--library-size', '-l', '-L', metavar='<max-library-size>',
        action='append',
        help='Library size to test. If none are provided, minimum and maximum possible are used.'
    )
    
    parser.add_argument(
        '--replicates', '-r', '-R', metavar='<n-reps>', type=int, default=100,
        help='Number of CCM replicates to run for each library size.'
    )
    
    parser.add_argument(
        '--embedding-dimension', '-E', metavar='<embedding-dimension>', type=int, default=None,
        help='Embedding dimension to use. If specified, -tau must be specified; these override -Emin and -Emax; no search is performed.'
    )
    
    parser.add_argument(
        '--embedding-lag', '-tau', metavar='<tau>', type=int, default=None,
        help='Embedding lag to use. If specified, -E must be specified; these override -taumin and -taumax; no search is performed.'
    )
    
    parser.add_argument(
        '--min-embedding-dimension', '-Emin', metavar='<min-embedding-dimension>', type=int, default=1,
        help='Minimum embedding dimension to test for identification.'
    )
    
    parser.add_argument(
        '--max-embedding-dimension', '-Emax', metavar='<max-embedding-dimension>', type=int, default=5,
        help='Embedding dimension used for prediction.'
    )
    
    parser.add_argument(
        '--min-embedding-lag', '-taumin', metavar='<tau>', type=int, default=1,
        help='Minimum time lag used to reconstruct attractor.'
    )
    
    parser.add_argument(
        '--max-embedding-lag', '-taumax', metavar='<tau>', type=int, default=5,
        help='Minimum time lag used to reconstruct attractor.'
    )
    
    parser.add_argument(
        '--max-cross-map-lag', '-xlag', metavar='<max-cross-map-lag>', type=int, default=5,
        help='Maximum cross-map lag.'
    )
    
    parser.add_argument(
        '--neighbor-count', '-K', '-k', metavar='<neighbor-count>', type=int, default=None,
        help='Number of neighbors to use. If unspecified, set to embedding dimension + 1.'
    )
    
    parser.add_argument(
        '--overwrite-output', '-o', action='store_true',
        help='Overwrite output file if it already exists.'
    )
    
    parser.add_argument(
        '--temporal-separation', '-dt', type=int, default=None,
        help='Minimum temporal separation to nearest-neighbor delay vectors. If unspecified, set to 3x the time at which autocorrelation reaches 1/e.'
    )
    
    parser.add_argument(
        '--cores', '-p', type=int, default=1,
        help='Number of cores to distribute analyses onto.'
    )
    
    parser.add_argument(
        '--bootstraps', '-b', '-B', type=int, default=100,
        help='Number of bootstrapped libraries to sample.'
    )
    
    return parser.parse_args()

def load_data(filename, vars, table_name):
    base, ext = os.path.splitext(filename)
    
    if ext.startswith('.sqlite'):
        data = load_data_sqlite(filename, vars, table_name)
    else:
        data = load_data_csv(filename, vars)
    
    # Check data length
    n_vals = None
    for var_name, values in data.iteritems():
        if n_vals is None:
            n_vals = values.shape[0]
        assert values.shape[0] == n_vals
    
    return data

def load_data_sqlite(filename, var_names, table_name):
    if not os.path.exists(filename):
        sys.stderr.write('{} does not exist; quitting.\n'.format(filename))
        sys.exit(1)
    if table_name is None:
        sys.stderr.write('No table specified; quitting.\n')
        sys.exit(1)
    
    db = sqlite3.connect(filename)
    
    if var_names is None:
        c = db.execute('SELECT * FROM {}'.format(table_name))
        var_names = [entry[0] for entry in c.description]
    
    data = OrderedDict()
    for var_name in var_names:
        values = []
        for row in db.execute('SELECT {} FROM {}'.format(var_name, table_name)):
            try:
                val = float(row[0])
            except:
                val = float('nan')
            values.append(val)
        
        if var_name in data:
            sys.stderr.write('Variable {} found twice; quitting.\n'.format(var_name))
            sys.exit(1)
        data[var_name] = numpy.array(values)
    
    db.close()
    
    return data

def load_data_csv(filename, var_names):
    if not os.path.exists(filename):
        sys.stderr.write('{} does not exist; quitting.\n'.format(filename))
        sys.exit(1)
    
    if var_names is None:
        with open(filename, 'rU') as f:
            cr = csv.reader(f)
            var_names = [x.strip() for x in cr.next() if x.strip() != '']
    
    data = OrderedDict()
    for var_name in var_names:
        with open(filename, 'rU') as f:
            values = []
            for row_dict in csv.DictReader(f):
                try:
                    val = float(row_dict[var_name])
                except:
                    val = float('nan')
                values.append(val)
            
            if var_name in data:
                sys.stderr.write('Variable {} found twice; quitting.\n'.format(var_name))
                sys.exit(1)
            data[var_name] = numpy.array(values)
    
    return data

if __name__ == '__main__':
    main()
