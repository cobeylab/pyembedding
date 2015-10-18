import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
import json
import jsonobject
import random
from math import sin, pi, log, exp, sqrt, floor, ceil, log1p, isnan

class ExecutionException(Exception):
    def __init__(self, cause, stdout_data, stderr_data):
        self.cause = cause
        self.stdout_data = stdout_data
        self.stderr_data = stderr_data

def run_via_pypy(model_name, params):
    # Prevent PyPy from trying to load CPython .pyc file
    pyc_filename = os.path.splitext(__file__)[0] + '.pyc'
    if os.path.exists(pyc_filename):
        os.remove(pyc_filename)

    import subprocess
    proc = subprocess.Popen(
        ['/usr/bin/env', 'pypy', '-B', os.path.abspath(__file__), model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False
    )

    if isinstance(params, jsonobject.JSONObject):
        stdin_data = params.dump_to_string()
    else:
        stdin_data = json.dumps(params)
    stdout_data, stderr_data = proc.communicate(stdin_data)
    proc.wait()

    try:
        return jsonobject.load_from_string(stdout_data)
    except Exception as e:
        raise ExecutionException(e, stdout_data, stderr_data)

def multistrain_sde(
    random_seed=None,
    dt_euler=None,
    adaptive=None,
    t_end=None,
    dt_output=None,
    n_pathogens=None,
    S_init=None,
    I_init=None,
    mu=None,
    nu=None,
    gamma=None,
    beta0=None,
    beta_change_start=None,
    beta_slope=None,
    psi=None,
    omega=None,
    eps=None,
    sigma=None,
    
    shared_proc=False,
    sd_proc=None,
    
    shared_obs=False,
    sd_obs=None,
    
    shared_obs_C=False,
    sd_obs_C=None,
    
    tol=None
):
    if random_seed is None:
        sys_rand = random.SystemRandom()
        random_seed = sys_rand.randint(0, 2**31 - 1)
    rng = random.Random()
    rng.seed(random_seed)
    
    pathogen_ids = range(n_pathogens)
    
    stochastic = sum([sd_proc[i] > 0.0 for i in pathogen_ids]) > 0
    assert not (adaptive and stochastic)
    
    has_obs_error = (sd_obs is not None) and (sum([sd_obs[i] > 0.0 for i in pathogen_ids]) > 0)
    has_obs_error_C = (sd_obs_C is not None) and (sum([sd_obs_C[i] > 0.0 for i in pathogen_ids]) > 0)
    
    log_mu = log(mu)
    log_gamma = [float('-inf') if gamma[i] == 0.0 else log(gamma[i]) for i in range(n_pathogens)]
    
    n_output = int(ceil(t_end / dt_output))
    
    def beta_t(t, pathogen_id):
        return (beta0[pathogen_id] + max(0.0, t - beta_change_start[pathogen_id]) * beta_slope[pathogen_id]) * (
            1.0 + eps[pathogen_id] * sin(
                2.0 * pi / psi[pathogen_id] * (t - omega[pathogen_id] * psi[pathogen_id])
            )
        )
    
    def step(t, h, logS, logI, CC):
        neg_inf = float('-inf')
        
        sqrt_h = sqrt(h)
        
        log_betas = [log(beta_t(t, i)) for i in pathogen_ids]
        try:
            logR = [log1p(-(exp(logS[i]) + exp(logI[i]))) for i in pathogen_ids]
        except:
            R = [max(0.0, 1.0 - exp(logS[i]) - exp(logI[i])) for i in pathogen_ids]
            logR = [neg_inf if R[i] == 0 else log(R[i]) for i in pathogen_ids]
        
        if stochastic:
            if shared_proc:
                noise = [rng.gauss(0.0, 1.0)] * n_pathogens
            else:
                noise = [rng.gauss(0.0, 1.0) for i in pathogen_ids]
            for i in pathogen_ids:
                noise[i] *= sd_proc[i]
        
        dlogS = [0.0 for i in pathogen_ids]
        dlogI = [0.0 for i in pathogen_ids]
        dCC = [0.0 for i in pathogen_ids]
        
        for i in pathogen_ids:
            dlogS[i] += (exp(log_mu - logS[i]) - mu) * h
            if gamma[i] > 0.0 and logR[i] > neg_inf:
                dlogS[i] += exp(log_gamma[i] + logR[i] - logS[i]) * h
            for j in pathogen_ids:
                if i != j:
                    dlogSRij = sigma[i][j] * exp(log_betas[j] + logI[j])
                    dlogS[i] -=  dlogSRij * h
                    if stochastic:
                        dlogS[i] -= dlogSRij * noise[j] * sqrt_h
            dlogS[i] -= exp(log_betas[i] + logI[i]) * h
            dlogI[i] += exp(log_betas[i] + logS[i]) * h
            dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * h
            if stochastic:
                dlogS[i] -= exp(log_betas[i] + logI[i]) * noise[i] * sqrt_h
                dlogI[i] += exp(log_betas[i] + logS[i]) * noise[i] * sqrt_h
                dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * noise[i] * sqrt_h
            dlogI[i] -= (nu[i] + mu) * h
            
        return [logS[i] + dlogS[i] for i in pathogen_ids], \
            [logI[i] + dlogI[i] for i in pathogen_ids], \
            [CC[i] + dCC[i] for i in pathogen_ids]
    
    logS = [log(S_init[i]) for i in pathogen_ids]
    logI = [log(I_init[i]) for i in pathogen_ids]
    CC = [0.0 for i in pathogen_ids]
    h = dt_euler
    
    
    ts = [0.0]
    logSs = [logS]
    logIs = [logI]
    CCs = [CC]
    Cs = [CC]
    
    if adaptive:
        sum_log_h_dt = 0.0
    for output_iter in range(n_output):
        min_h = h
        
        t = output_iter * dt_output
        t_next_output = (output_iter + 1) * dt_output
        
        while t < t_next_output:
            if h < min_h:
                min_h = h
            
            t_next = t + h
            if t_next > t_next_output:
                t_next = t_next_output
            logS_full, logI_full, CC_full = step(t, t_next - t, logS, logI, CC)
            if adaptive:
                t_half = t + (t_next - t)/2.0
                logS_half, logI_half, CC_half = step(t, t_half - t, logS, logI, CC)
                logS_half2, logI_half2, CC_half2 = step(t_half, t_next - t_half, logS_half, logI_half, CC_half)
                
                errorS = [logS_half2[i] - logS_full[i] for i in pathogen_ids]
                errorI = [logI_half2[i] - logI_full[i] for i in pathogen_ids]
                errorCC = [CC_half2[i] - CC_full[i] for i in pathogen_ids]
                max_error = max([abs(x) for x in (errorS + errorI + errorCC)])
                
                if max_error > 0.0:
                    h = 0.9 * (t_next - t) * tol / max_error
                else:
                    h *= 2.0
                
                if max_error < tol:
                    sum_log_h_dt += (t_next - t) * log(t_next - t)
                    
                    logS = [logS_full[i] + errorS[i] for i in pathogen_ids]
                    logI = [logI_full[i] + errorI[i] for i in pathogen_ids]
                    CC = [CC_full[i] + errorCC[i] for i in pathogen_ids]
                    t = t_next
            else:
                logS = logS_full
                logI = logI_full
                CC = CC_full
                t = t_next
        ts.append(t)
        logSs.append(logS)
        if not has_obs_error:
            logIs.append(logI)
        else:
            if shared_obs:
                obs_err = rng.gauss(0.0, 1.0)
                obs_errs = [obs_err * sd_obs[i] for i in pathogen_ids]
            else:
                obs_errs = [rng.gauss(0.0, sd_obs[i]) for i in pathogen_ids]
            logIs.append([logI[i] + obs_errs[i] for i in pathogen_ids])
        CCs.append(CC)
        if has_obs_error_C:
            if shared_obs_C:
                obs_err = rng.gauss(0.0, 1.0)
                obs_errs = [obs_err * sd_obs_C[i] for i in pathogen_ids]
            else:
                obs_errs = [rng.gauss(0.0, sd_obs_C[i]) for i in pathogen_ids]
        else:
            obs_errs = [0.0 for i in pathogen_ids]
        Cs.append([max(0.0, CCs[-1][i] - CCs[-2][i] + obs_errs[i]) for i in pathogen_ids])

    result = jsonobject.JSONObject([
        ('t', ts),
        ('logS', logSs),
        ('logI', logIs),
        ('C', Cs),
        ('random_seed', random_seed)
    ])
    if adaptive:
        result.dt_euler_harmonic_mean = exp(sum_log_h_dt / t)
    return result

if __name__ == '__main__':
    params = json.load(sys.stdin)
    result = globals()[sys.argv[1]](**params)
    result.dump_to_file(sys.stdout)
