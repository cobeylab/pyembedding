# pyembedding

Ed Baskerville
last update: 26 February 2015

NB: Usage instructions out of date but repo is published for the sake of reproducibility for paper.

## Overview

This package contains a Python implementation of nonlinear time-series embedding methods.

## Installation

Requires Python 2.7.x with numpy and scipy.

Recommended way to install Python:
* [Anaconda distribution](http://continuum.io/downloads)

After you have installed Anaconda and configured your `PATH` environment variable, make
sure that

```{sh}
$ python --version
```

includes the word "Anaconda" in the description.

## ccm.py

The script `ccm.py` automates the method of convergent cross-mapping (TODO: ref Sugihara et al.).

Run
```{sh}
ccm.py --help
```
to see detailed usage information.

Example:
```{sh}
ccm.py --n-cores 4 -E 4 -t 1 -L "5:50:5" -R 100 -C "x,y" -V "x:y,y:x" input.sqlite timeseries output.sqlite
```
will read time-series data from columns `x` and `y` (`-C "x,y"`) in table `timeseries`
in the SQLite database `input.sqlite`; test causality for `x` causing `y` as well as
`y` causing `x`; and write output to the file `output.sqlite`; using these parameters:
* embedding dimension: 4 (`-E 4`)
* lag (tau): 1 (`-t 1`)
* library lengths: 5, 10, ..., 45, 50 (`-L "5:50:5"`)
* replicate samples per library length: 100 (`-R 100`)

The output database will contain two tables:
* `args`: contains the arguments used to run the program
* `results`: contains analysis results.

The `results` table contains these columns:
* `cause`: the variable being tested as cause
* `effect`: the variable being tested as effect
* `L`: the library length being tested
* `replicate_id`: unique ID for this replicate
* `corr`: the correlation between the original "cause" time series and the version
  reconstructed from lagged vectors sampled from "effect".

## API

TODO
