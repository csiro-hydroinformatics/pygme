#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2022-01-31 14:50:03.285909
## Comment : Calibrate Sacramento using given param lib
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs, fmin

from hydrodiy.io import csv, iutils
from hydrodiy.io.hyruns import get_batch, OptionManager
from hydrodiy.stat import metrics, sutils, transform

from pygme.factory import calibration_factory
from pygme.factory import model_factory
from pygme.factory import parameters_transform_factory
from pygme.factory import objfun_factory

from pygme import calibration

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="A script", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-v", "--version", \
                    help="Version number", \
                    type=int, required=True)
parser.add_argument("-t", "--taskid", help="Task id", \
                    type=int, required=True)
parser.add_argument("-n", "--nbatch", help="Number of batch processes", \
                    type=int, default=20)
parser.add_argument("-o", "--overwrite", help="Overwrite", \
                    action="store_true", default=False)
parser.add_argument("-d", "--debug", help="Debug mode", \
                    action="store_true", default=False)
args = parser.parse_args()

taskid = args.taskid
version = args.version
overwrite = args.overwrite
debug = args.debug
nbatch = args.nbatch

# Get option manager
opm = OptionManager()
models = ["HBV"] #["IHACRES", "SAC15", "GR6J", "WAPABA"]
objfuns = ["bc0.2", "biasbc0.2", "bc0.5", "biasbc0.5", \
                                "bc1.0", "biasbc1.0"]
batches = np.arange(nbatch)
opm.from_cartesian_product(model=models, objfun=objfuns, batch=batches)

# Get task
task = opm.get_task(taskid)
model_name = task.model
batch = task.batch
objfun_name = task.objfun

# Large params lib
nparamslib = 20000 if model_name == "SAC15" else 5000

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fdata = froot.parent.parent.parent / "Data"\
    / "characterisation_paper" / "export"

fout = froot / "outputs" / f"calmodel_{model_name}_v{version}"
fout.mkdir(exist_ok=True, parents=True)

#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
flogs = froot / "logs" / "calmodel"
flogs.mkdir(exist_ok=True, parents=True)
basename = source_file.stem
flog = flogs / f"calmodel_TASK{taskid}_V{version}.log"
LOGGER = iutils.get_logger(basename, console=debug, flog=flog)

LOGGER.info(f"version: {version}")
LOGGER.info(f"taskid: {taskid}")
LOGGER.info(f"model: {model_name}")
LOGGER.info(f"objfun: {objfun_name}")
LOGGER.info(f"batch:  {batch}")

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
LOGGER.info("Loading data")
dversion = "5.0"
fs = fdata / f"stations_v{dversion}.csv"
sites, _ = csv.read_csv(fs, index_col="STATIONID")
sites = sites.loc[sites.IS_VALID == 1]

# select site from batch
isites = get_batch(len(sites), nbatch, batch)
sites = sites.iloc[isites, :]

if debug:
    sites = sites.iloc[:2]

fd = fdata / "daily.parquet_v5.0.gzip"
daily = pd.read_parquet(fd)
isin = daily.STATIONID.isin(sites.index)
daily = daily.loc[isin]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
timestep = "MS" if model_name in ["GR2M", "IHACRES"] else "D"

# Load model objects
true2trans, trans2true = parameters_transform_factory(model_name)

# Reciprocal transform for NSE reciprocal
trans_recip = transform.Reciprocal()
trans_recip.nu = 1e-1 if timestep=="MS" else 1e-3

# 10 years of warmup
warmup = int(10*365.25) if timestep == "D" else 120

# Objective function and calibration object
objfun = objfun_factory(objfun_name)
calib = calibration_factory(model_name, \
                    objfun=objfun, \
                    nparamslib=nparamslib, \
                    warmup=warmup)
model = calib.model

# Load parameters
if version == 1:
    LOGGER.info(f"Retrieve parameter library from package")
    plib = calib.paramslib
    tplib = plib*0.
    for i, p in enumerate(plib):
        tplib[i] = true2trans(p)

    means = tplib.mean(axis=0)
    cov = np.cov(tplib.T)

else:
    LOGGER.info(f"Retrieve parameter library from version {version-1}")
    flib = fout.parent / f"calmodel_{model_name}_v{version-1}"
    lf = list(flib.glob(f"params_*{model_name}_*_v{version-1}.json"))
    nf = len(lf)
    tplib = []
    perfs = []
    for i, f in enumerate(f):
        with f.open("r") as fo:
            p = json.load(fo)
        nse = p["nse"]
        bias = p["bias"]
        perfs.append([nse, abs(bias)])

        # Discard parameter with very low perf
        if nse>0 and abs(bias)<0.5:
            pv = np.array([p["params"][n] for n in model.params.names])
            tplib.append(true2trans(pv))

    tplib = np.array(tplib)
    means = np.mean(tplib, axis=0)
    cov = np.cov(tplib.T)

    if "Sarva" in model.params.names:
        isarva = np.where(model.params.names == "Sarva")[0]
        cov[isarva, isarva] = 0.1

# Build parameter library from
# MVT norm in transform space using latin hypercube
tplib = sutils.lhs_norm(nparamslib, means, cov)

# Back transform
plib = tplib * 0.
for i in range(len(plib)):
    plib[i, :] = trans2true(tplib[i, :])
plib = np.clip(plib, model.params.mins, model.params.maxs)

# Set library
calib.paramslib = plib

# Run calibration
nsites = len(sites)
for i, (stationid, row) in enumerate(sites.iterrows()):
    LOGGER.info(f"dealing with {stationid} ({i+1}/{nsites})")

    fparams = fout / f"params_{stationid}_{model_name}_o{objfun_name}_v{version}.json"
    if fparams.exists() and not overwrite:
        LOGGER.info("... file already exists. Skip")
        continue

    # Get data
    data = daily.loc[daily.STATIONID == stationid]
    dates = data.index
    obs = data.Q.values.squeeze()
    obs[obs<0] = np.nan

    inputs = data.loc[:, ["RAIN", "PET"]].values

    # Select data with available flow + warmup
    nval = len(data)
    start = max(0, np.where(~np.isnan(obs))[0].min()-warmup)
    end = min(nval-1, np.where(~np.isnan(obs) & ~np.isnan(inputs[:, 0]))[0].max())
    date_start = dates[start]
    date_end = dates[end]
    obs = obs[start:end]
    inputs = np.ascontiguousarray(inputs[start:end])

    # Set calibration period with warmup
    obs[:warmup] = np.nan
    nval = len(obs)
    ical = np.where(~np.isnan(obs) & (np.arange(len(obs))>warmup))[0]

    min_length = 3650 if timestep == "D" else 120
    if len(ical) < min_length:
        LOGGER.error(f"Record too short ({len(ical)} {timestep}). Skip")
        continue

    # Calibrate
    #final, ofun, sfinal, oexplo = calib.workflow(obs, inputs, ical=ical,
    #                                             optimizer=fmin)

    # Detailed calib process
    calib.allocate(obs, inputs)
    calib.ical = ical
    start, _, ofuns = calib.explore(iprint=500)
    sys.exit()

    #final, _, _ = cal.fit(iprint=10)

    # Compute simple perfs NSE / bias
    model = calib.model
    o, s = obs[ical], model.outputs[ical, 0]
    nse = metrics.nse(o, s)
    nserecip = metrics.nse(o, s, trans=trans_recip)
    bias = metrics.bias(o, s)

    sys.exit()


    # Store
    params = {n:round(v, 3) for n, v in zip(model.params.names, final)}
    dd = {
        "stationid": stationid,
        "version": version,
        "model": model_name,
        "objfun": objfun_name,
        "warmup": warmup,
        "nparamslib": nparamslib,
        "nse": round(nse, 3),
        "nserecip": round(nserecip, 3),
        "bias": round(bias, 3),
        "params": params
    }
    with fparams.open("w") as fo:
        json.dump(dd, fo, indent=4)


LOGGER.info("Process completed")

