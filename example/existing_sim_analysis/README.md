# Analysis scripts for existing simulations

The scripts in this directory are used to perform CCM analyses for simulations already
present in an output database (see example/simulation). The following sections describe
the steps required to run an analysis.

## Get the code

```{sh}
mkdir 2015-11-11-analysis
cd 2015-11-11-analysis
git clone git@bitbucket.org:cobeylab/pyembedding.git
cp pyembedding/example/existing_sim_analysis/* .
```

## Modify generate_jobs.py

Modify `generate_jobs.py` to contain desired settings. Make sure `simulation_db_path` is set to the
actual absolute path of the simulation database.

## Modify generate_jobs.py for a test run

Modify generate_jobs.py with `n_replicates = 1` as a test. This will ensure that
only the first replicate from each parameter combination will be used from the simulation
database.

## Run generate_jobs.py

```{sh}
./generate_jobs.py
```

This will create a directory called `jobs` containing subdirectories for each simulation
run; these are used as working directories for the analysis.

## Test a single job manually

Using a single processor, manually use the `run_job.py` script to make sure things work
for a single run. E.g.:

```
sinteractive
cd jobs/eps=0.0-beta00=0.30-sigma01=1.00-sd_proc=0.100/000
../../../run_job.py
```

The `run_job.py` script knows to load information from the `runmany_info.json` file in the
current working directory, so it doesn't need any arguments; it just needs to run
in the right place.

## Test the 1-replicate sweep

Use `runmany` to run a sweep:

```{sh}
runmany slurm ccm jobs chunks
```

## Gather jobs together

```{sh}
gather jobs results_gathered.sqlite
```

## Set up the real thing

Remove the test `jobs` and `chunks` directories:

```{sh}
rm -r jobs
rm -r chunks
rm results_gathered.sqlite
```

and then modify `generate_jobs.py` to have `N_REPLICATES = 100`.

Then, re-run things for real:

```{sh}
./generate_jobs.py
runmany slurm ccm jobs chunks
gather jobs results_gathered.sqlite
```
