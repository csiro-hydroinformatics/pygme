#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2022-02-02 Wed 02:38 PM
## Comment : Check params libraries
##
## ------------------------------

import sys, os, re, json, math
import argparse
from pathlib import Path

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from pygme.models import sac15

from tqdm import tqdm

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="A script", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-p", "--progress", help="Show progress", \
                    action="store_true", default=False)
parser.add_argument("-o", "--overwrite", help="Overwrite existing param data",\
                    action="store_true", default=False)
args = parser.parse_args()

progress = args.progress
overwrite = args.overwrite

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent
fout = froot / "outputs"

fimg = froot / "images"
fimg.mkdir(exist_ok=True)

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
fconcat = fout / "calsac_params.csv"

if fconcat.exists() and not overwrite:
    params, _ = csv.read_csv(fconcat)
else:
    lf = list(fout.glob("**/sacparams*.json"))
    nf = len(lf)
    tbar = tqdm(enumerate(lf), desc="Loading params", \
                    total=nf, disable=not progress)
    params = []
    for i, f in tbar:
        with f.open("r") as fo:
            p = json.load(fo)

        pp = p["params"].copy()
        for cn in p.keys():
            if cn == "params":
                continue
            pp[cn] = p[cn]
        params.append(pp)

    params = pd.DataFrame(params)

    csv.write_csv(params, fconcat, \
        "SAC params", source_file, \
        compress=False)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Finalise parameter lib configuration
versions = params.version.unique()
version = versions.max()

model = sac15.SAC15()
pnames = model.params.names
plib = params.loc[params.version == version, pnames]
tplib = plib*0.
for i, p in plib.iterrows():
    tplib.loc[i, :] = sac15.sac15_true2trans(p.values)

means = tplib.mean()
cov = tplib.cov()

f = fout / "sacparamslib.txt"
with f.open("w") as fo:
    fo.write(f"--- Paramslib version {version} ---\n\n")

    txt = " ".join([f"{pn:s}" for pn in pnames])
    fo.write("Parameters: \n" + txt + "\n\n")

    txt = ", ".join([f"{v:0.2f}" for v in means.values])
    fo.write("Transformed means: \n[" + txt + "]\n\n")

    txt = ""
    for co in cov.values:
        line = ", ".join([f"{v:0.2f}" for v in co])
        txt += "\t[" + line + "],\n"
    fo.write("Transformed covariances: \n[" + txt + "]\n\n")

# Plot
plt.close("all")
fig, axs = plt.subplots(nrows=3, ncols=6,\
                    figsize=(15, 10), \
                    constrained_layout=True)

cols = [cn for cn in params.columns \
            if not re.search("siteid|version|warmup|nparamslib", cn)]

for iplot, (ax, cn) in enumerate(zip(axs.flat, cols)):
    df = pd.pivot_table(params.loc[:, [cn, "siteid", "version"]], \
                    columns="version", index="siteid", \
                    values=cn)
    if cn == "bias":
        df = np.abs(df)

    putils.ecdfplot(ax, df)

    if cn == "nse":
        ax.set_xlim((0.2, 0.9))
    elif cn == "bias":
        ax.set_xlim((0, 0.3))
    elif cn == "Lag":
        ax.set_xlim((0, 10))

    if iplot == 0:
        ax.legend(loc=4, fontsize="x-small")
    ax.text(0.05, 0.95, cn, transform=ax.transAxes, fontweight="bold")

fp = fimg / "calsac_version.png"
fig.savefig(fp)

LOGGER.info("Process completed")

