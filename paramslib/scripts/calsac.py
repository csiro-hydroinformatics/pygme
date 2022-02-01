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

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import metrics, sutils

from datasets import Dataset

from pygme.models import sac15

import importlib
importlib.reload(sac15)


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
parser.add_argument("-s", "--sitepattern", help="Site selection pattern", \
                    type=str, default="")
parser.add_argument("-i", "--ibatch", help="Batch process number", \
                    type=int, default=-1)
parser.add_argument("-n", "--nbatch", help="Number of batch processes", \
                    type=int, default=20)
parser.add_argument("-p", "--progress", help=" Show progress", \
                    action="store_true", default=False)
parser.add_argument("-o", "--overwrite", help="Overwrite", \
                    action="store_true", default=False)
args = parser.parse_args()

version = args.version
ibatch = args.ibatch
nbatch = args.nbatch
progress = args.progress
sitepattern = args.sitepattern
overwrite = args.overwrite

# 10 years of warmup
warmup = int(10*365.25)

# Large params lib
nparamslib = 20000

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent
fout = froot / "outputs" / f"calsac_v{version}"
fout.mkdir(exist_ok=True, parents=True)

flogs = froot / "logs" / "calsac"
flogs.mkdir(exist_ok=True, parents=True)
basename = source_file.stem
flog = flogs / f"calsac_TASK{ibatch}_V{version}.log"
LOGGER = iutils.get_logger(basename, flog=flog)

LOGGER.info(f"version: {version}")
LOGGER.info(f"ibatch:  {ibatch}")

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
dset = Dataset("OZDATA", "1.0")
sites_all = dset.get_sites()

# Select sites
sites = sites_all
if not sitepattern == "":
    idx = sites_all.index.str.findall(sitepattern).astype(bool)
    sites = sites_all.loc[idx, :]
else:
    if ibatch >=0:
        idx = hyruns.get_batch(sites_all.shape[0], nbatch, ibatch)
        sites = sites_all.iloc[idx, :]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

#load params lib from previous versions
model = sac15.SAC15()
if version == 1:
    LOGGER.info(f"Retrieve parameter library from package")
    means = sac15.SAC15_TMEAN
    cov = sac15.SAC15_TCOV
else:
    LOGGER.info(f"Retrieve parameter library from version {version-1}")
    flib = fout.parent / f"calsac_v{version-1}"
    lf = list(flib.glob(f"sacparams*v{version-1}.json"))
    nf = len(lf)
    tplib = np.zeros((nf, model.params.nval))
    #perf = np.zeros((nf, 2))
    tbar = tqdm(enumerate(lf), desc="Loading params", \
                    total=nf, disable=not progress)
    for i, f in tbar:
        with f.open("r") as fo:
            p = json.load(fo)
        pv = np.array([p["params"][n] for n in model.params.names])
        tplib[i, :] = sac15.sac15_true2trans(pv)
        #perf[i, :] = [p["nse"], p["bias"]]

    means = np.mean(tplib, axis=0)
    cov = np.cov(tplib.T)

# Build parameter library from
# MVT norm in transform space using latin hypercube
tplib = sutils.lhs_norm(nparamslib, means, cov)

# Back transform
plib = tplib * 0.
for i in range(len(plib)):
    plib[i, :] = sac15.sac15_trans2true(tplib[i, :])
plib = np.clip(plib, model.params.mins, model.params.maxs)

# Run calibration
tbar = tqdm(enumerate(sites.iterrows()), total=len(sites), \
                desc="Sac", disable=not progress)
for i, (siteid, row) in tbar:
    if not progress:
        LOGGER.info("dealing with {0} ({1}/{2})".format( \
            siteid, i, len(sites)))

    fparams = fout / f"sacparams_{siteid}_v{version}.json"
    if fparams.exists() and not overwrite:
        continue

    # Get data
    daily, _ = dset.get(siteid, "rainfall_runoff", "D")
    dates = daily.index
    obs = daily.loc[:, "runoff[mm/d]"].values
    inputs = daily.loc[:, ["rain[mm/d]", "evap[mm/d]"]].values

    start = max(0, np.where(~np.isnan(obs))[0].min()-warmup)
    end = min(len(daily)-1, np.where(~np.isnan(obs))[0].max())
    date_start = dates[start]
    date_end = dates[end]
    obs = obs[start:end]
    inputs = np.ascontiguousarray(inputs[start:end])
    ical = np.where(~np.isnan(obs) & (np.arange(len(obs))>warmup))[0]

    if len(ical) < 365*10:
        LOGGER.error(f"Record too short ({len(ical)} days). Skip")
        continue

    # Calibrate on whole period
    cal = sac15.CalibrationSAC15(warmup=warmup, nparamslib=nparamslib)

    # Set paramslib
    cal.paramslib = plib

    # Calibrate
    final, _, _, _ = cal.workflow(obs, inputs, ical=ical)

    # Compute simple perfs NSE / bias
    o, s = obs[ical], cal.model.outputs[ical, 0]
    nse = metrics.nse(o, s)
    bias = metrics.bias(o, s)

    # Store
    params = {n:round(v, 3) for n, v in zip(cal.model.params.names, final)}
    dd = {
        "siteid": siteid, \
        "version": version, \
        "warmup": warmup, \
        "nparamslib": nparamslib, \
        "nse": round(nse, 3), \
        "bias": round(bias, 3), \
        "params": params
    }
    with fparams.open("w") as fo:
        json.dump(dd, fo, indent=4)

LOGGER.info("Process completed")

