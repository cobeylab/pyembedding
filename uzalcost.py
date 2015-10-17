import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
import tempfile
import subprocess
import numpy
from collections import OrderedDict
import shutil

def configure(dir):
    cwd = os.getcwd()
    
    os.chdir(dir)
    proc = subprocess.Popen(['./configure'])
    result = proc.wait()
    if result != 0:
        raise Exception('configure returned nonzero status')
    
    os.chdir(cwd)

def make(dir):
    proc = subprocess.Popen(['make'], cwd=dir)
    result = proc.wait()
    if result != 0:
        raise Exception('make returned nonzero status')

def run_and_load_files(args, stdin_data, filenames):
    tmp_dir = tempfile.mkdtemp()
    
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        cwd=tmp_dir
    )
    stdout_data, stderr_data = proc.communicate(stdin_data)
    
    file_data_dict = OrderedDict()
    for filename in filenames:
        filepath = os.path.join(tmp_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                file_data = f.read()
            file_data_dict[filename] = file_data
    
    shutil.rmtree(tmp_dir)
    
    return stdout_data, stderr_data, file_data_dict

def parse_results(file_data):
    ms = []
    Lks = []
    for line in file_data.split('\n'):
        if not line.startswith('#') and len(line) > 0:
            try:
                pieces = line.split(' ')
                m = int(pieces[0]) + 1
                Lk = float(pieces[1])
                ms.append(m)
                Lks.append(Lk)
            except:
                pass
    
    return ms, Lks

def parse_params(stderr_data):
    params = {}
    for line in stderr_data.split('\n'):
        if line.startswith('Using T_M='):
            pieces = line.split('=')
            params['tw_max'] = int(pieces[1])
        elif line.startswith('Using ThW='):
            pieces = line.split('=')
            params['theiler_window'] = int(pieces[1])
        elif line.startswith('Using k='):
            pieces = line.split('=')
            params['n_neighbors'] = int(pieces[1].split(' ')[0])
    return params

def set_up_uzal_costfunc():
    uzal_dir = os.path.join(SCRIPT_DIR, 'optimal_embedding')
    costfunc_path = os.path.join(uzal_dir, 'source_c', 'costfunc')
    if not os.path.exists(costfunc_path):
        configure(uzal_dir)
        make(uzal_dir)
    
    return costfunc_path
        

def run_uzal_costfunc(x, neighbor_count=None, theiler_window=None, max_prediction_horizon=None, max_window=None):
    r'''Runs the Uzal et al. cost function for full embeddings.
    >>> ms, Lks, params = run_uzal_costfunc(numpy.random.normal(0, 1, size=1000))

    >>> sys.stderr.write('ms = {0}\n'.format(ms))
    >>> sys.stderr.write("Lks = {0}\n".format(Lks))
    >>> sys.stderr.write("params = {0}\n".format(params))
    '''
    costfunc_path = set_up_uzal_costfunc()

    stdin_data = '\n'.join(['{0}'.format(xi) for xi in x])
    stdin_data += '\n'

    args = [costfunc_path, '-e', '2']
    if neighbor_count is not None:
        args += ['-k', str(neighbor_count)]
    if theiler_window is not None:
        args += ['-t', str(theiler_window)]
    if max_prediction_horizon is not None:
        args += ['-s', str(max_prediction_horizon)]
    if max_window is not None:
        args += ['-W', str(max_window)]

    stdout_data, stderr_data, file_data_dict = run_and_load_files(
        args, stdin_data,
        ['stdin.amp']
    )
    
    sys.stderr.write(stdout_data)
    sys.stderr.write(stderr_data)
    
    ms, Lks = parse_results(file_data_dict['stdin.amp'])
    params = parse_params(stderr_data)
    
    return ms, Lks, params
