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

from datasets import Dataset

from pygme.models import sac15, wapaba, gr6j
from pygme import calibration

import importlib
importlib.reload(calibration)
importlib.reload(sac15)
importlib.reload(wapaba)
importlib.reload(gr6j)

from tqdm import tqdm

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
parser.add_argument("-p", "--progress", help=" Show progress", \
                    action="store_true", default=False)
parser.add_argument("-o", "--overwrite", help="Overwrite", \
                    action="store_true", default=False)
args = parser.parse_args()

taskid = args.taskid
version = args.version
progress = args.progress
overwrite = args.overwrite
nbatch = args.nbatch

# Large params lib
nparamslib = 20000

# Get option manager
opm = OptionManager()
models = ["sac15", "gr6j", "wapaba"]
batches = np.arange(nbatch)
opm.from_cartesian_product(model=models, batch=batches)

# Get task
task = opm.get_task(taskid)
model_name = task.model
batch = task.batch

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent
fout = froot / "outputs" / f"calmodel_{model_name}_v{version}"
fout.mkdir(exist_ok=True, parents=True)

flogs = froot / "logs" / "calmodel"
flogs.mkdir(exist_ok=True, parents=True)
basename = source_file.stem
flog = flogs / f"calmodel_TASK{taskid}_V{version}_M{model_name}_B{batch}.log"
LOGGER = iutils.get_logger("pygme.calibration", flog=flog)

LOGGER.info(f"taskid: {taskid}")
LOGGER.info(f"model: {model_name}")
LOGGER.info(f"version: {version}")
LOGGER.info(f"batch:  {batch}")

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
dset = Dataset("OZDATA", "1.0")
sites = dset.get_sites()

# select site from batch
isites = get_batch(len(sites), nbatch, batch)
sites = sites.iloc[isites, :]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Load model objects
if model_name == "sac15":
    model = sac15.SAC15()
    CalibrationObject = sac15.CalibrationSAC15
    means = sac15.SAC15_TMEAN
    cov = sac15.SAC15_TCOV
    true2trans = sac15.sac15_true2trans
    trans2true = sac15.sac15_trans2true
    timestep = "D"

elif model_name == "gr6j":
    model = gr6j.GR6J()
    CalibrationObject = gr6j.CalibrationGR6J
    means = gr6j.GR6J_TMEAN
    cov = gr6j.GR6J_TCOV
    true2trans = gr6j.gr6j_true2trans
    trans2true = gr6j.gr6j_trans2true
    timestep = "D"

elif model_name == "wapaba":
    model = wapaba.WAPABA()
    CalibrationObject = wapaba.CalibrationWAPABA
    means = wapaba.WAPABA_TMEAN
    cov = wapaba.WAPABA_TCOV
    true2trans = wapaba.wapaba_true2trans
    trans2true = wapaba.wapaba_trans2true
    timestep = "MS"

# Reciprocal transform
trans_recip = transform.Reciprocal()
trans_recip.nu = 1e-1 if timestep=="MS" else 1e-3

# 10 years of warmup
warmup = int(10*365.25) if timestep == "D" else 120

# Load parameters
if version == 1:
    LOGGER.info(f"Retrieve parameter library from package")
else:
    LOGGER.info(f"Retrieve parameter library from version {version-1}")
    flib = fout.parent / f"calmodel_m{model_name}_v{version-1}"
    lf = list(flib.glob(f"params_*_m{model_name}_v{version-1}.json"))
    nf = len(lf)
    tplib = []
    tbar = tqdm(enumerate(lf), desc="Loading params", \
                    total=nf, disable=not progress)
    for i, f in tbar:
        with f.open("r") as fo:
            p = json.load(fo)
        nse = p["nse"]
        bias = p["bias"]

        # Discard parameter with very low perf
        if nse>0 and abs(bias)<0.5:
            pv = np.array([p["params"][n] for n in model.params.names])
            tplib.append(true2trans(pv))

    tplib = np.array(tplib)
    means = np.mean(tplib, axis=0)
    cov = np.cov(tplib.T)

# Build parameter library from
# MVT norm in transform space using latin hypercube
tplib = sutils.lhs_norm(nparamslib, means, cov)

# Back transform
plib = tplib * 0.
for i in range(len(plib)):
    plib[i, :] = trans2true(tplib[i, :])
plib = np.clip(plib, model.params.mins, model.params.maxs)

# Run calibration
tbar = tqdm(enumerate(sites.iterrows()), total=len(sites), \
                desc="Calib", disable=not progress)
for i, (siteid, row) in tbar:
    if not progress:
        LOGGER.info("dealing with {0} ({1}/{2})".format( \
            siteid, i, len(sites)))

    fparams = fout / f"params_{siteid}_m{model_name}_v{version}.json"
    if fparams.exists() and not overwrite:
        continue

    # Get data
    data, _ = dset.get(siteid, "rainfall_runoff", timestep)
    dates = data.index
    obs = data.filter(regex="^runoff\[", axis=1).values.squeeze()
    inputs = data.filter(regex="^(evap|rain)\[", axis=1).values.squeeze()

    start = max(0, np.where(~np.isnan(obs))[0].min()-warmup)
    end = min(len(data)-1, np.where(~np.isnan(obs))[0].max())
    date_start = dates[start]
    date_end = dates[end]
    obs = obs[start:end]
    inputs = np.ascontiguousarray(inputs[start:end])
    ical = np.where(~np.isnan(obs) & (np.arange(len(obs))>warmup))[0]

    min_length = 3650 if timestep == "D" else 120
    if len(ical) < min_length:
        LOGGER.error(f"Record too short ({len(ical)} {timestep}). Skip")
        continue

    # Calibrate on whole period
    cal = CalibrationObject(warmup=warmup, nparamslib=nparamslib)

    # Set paramslib
    cal.paramslib = plib

    # Calibrate
    final, _, _, _ = cal.workflow(obs, inputs, ical=ical, \
                                    optimizer=fmin)

    #cal.allocate(obs, inputs)
    #cal.ical = ical
    #start, _, ofuns = cal.explore(iprint=500)
    #final, _, _ = cal.fit(iprint=10)

    # Compute simple perfs NSE / bias
    o, s = obs[ical], cal.model.outputs[ical, 0]
    nse = metrics.nse(o, s)
    nserecip = metrics.nse(o, s, trans=trans_recip)
    bias = metrics.bias(o, s)

    # Store
    params = {n:round(v, 3) for n, v in zip(cal.model.params.names, final)}
    dd = {
        "siteid": siteid, \
        "version": version, \
        "model": model_name, \
        "warmup": warmup, \
        "nparamslib": nparamslib, \
        "nse": round(nse, 3), \
        "nserecip": round(nserecip, 3), \
        "bias": round(bias, 3), \
        "params": params
    }
    with fparams.open("w") as fo:
        json.dump(dd, fo, indent=4)


LOGGER.info("Process completed")

