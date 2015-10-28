# Analysis scripts for existing simulations

The scripts in this directory are used to perform CCM analyses for simulations already
present in an output database (see example/simulation). The following sections describe
the steps required to run an analysis.

## Get the code

```{sh}
mkdir <EXPERIMENT-DIR>
cd <EXPERIMENT-DIR>
git clone git@bitbucket.org:cobeylab/pyembedding.git
cp pyembedding/example/analysis/*.py .
```

## Modify run_job.py

Modify run_job.py to contain desired settings. Make sure `SIM_DB_PATH` is set to the
actual absolute path of the simulation database.

## Modify generate_jobs.py for a test run

First make sure `SIM_DB_PATH` is set correctly in generate_jobs.py

Then modify generate_jobs.py so that N_REPLICATES = 1 as a test. This will ensure that
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

The `run_job.py` script knows to load information from the `job_info.json` file in the
current working directory, so it doesn't need any arguments; it just needs to run
in the right place.

## Modify generate_chunks.py

Modify generate_chunks.py to have correct SLURM settings, if necessary. The walltime limit
is probably the only thing that needs to be changed.

## Test the 1-replicate sweep

First, run `generate_chunks.py`. This creates a `chunks` directory with 16 jobs per chunk;
a SLURM `.sbatch` file is created in each subdirectory.

Then, run `submit_chunks.py` to actually submit the SLURM jobs.

```{sh}
./generate_chunks.py
./submit_chunks.py
./check_status.py
```

You can use

```{sh}
squeue -i 5 -u <username>
```

to query SLURM every 5 seconds about job status. You can also run

```
./check_status.py
```

to see a readout of how many individual jobs (within chunks) are waiting, running,
complete, or failed.

## Gather jobs together

To combine all output files into a single

## Set up the real thing

Remove the test `jobs` and `chunks` directories:

```{sh}
rm -rf jobs
rm -rf chunks
```

and then modify `generate_jobs.py` to have `N_REPLICATES = 100`.

Then, re-run things for real:

```{sh}
./generate_jobs.py
./generate_chunks.py
./submit_chunks.py
[WAIT FOR COMPLETION]
./gather.py
```


## Caveats

The number of simulations this was set up for happens to be exactly 16 * the number of
SLURM jobs we're allowed to submit, so no cleverness was included to add extra runs
to a SLURM job. If we make more than 8000 simulations, we'll need to do that.

The `SIM_DB_PATH` must be set in both `generate_jobs.py` and `run_job.py`.
