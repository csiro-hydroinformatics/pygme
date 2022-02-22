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

from pygme.models import sac15, gr6j, wapaba, ihacres

from tqdm import tqdm

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="A script", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-o", "--overwrite", help="Overwrite existing param data",\
                    action="store_true", default=False)
args = parser.parse_args()

overwrite = args.overwrite

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent
fout = froot / "outputs"
flibs = fout / "paramslib"
flibs.mkdir(exist_ok=True)

fimg = froot / "images"
fimg.mkdir(exist_ok=True)

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

for model_name in ["sac15", "gr6j", "wapaba"]:
    LOGGER.info(f"Processing {model_name}")

    #----------------------------------------------------------------------
    # Get data
    #----------------------------------------------------------------------
    fconcat = flibs / f"calmodel_params_{model_name}.csv"

    if fconcat.exists() and not overwrite:
        params, _ = csv.read_csv(fconcat)
    else:
        lf = list(fout.glob(f"**/params_*_{model_name}_*.json"))
        nf = len(lf)

        tbar = tqdm(enumerate(lf), desc=model_name, total=nf)
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
            f"{model_name} params", source_file, \
            compress=False)

    #----------------------------------------------------------------------
    # Process
    #----------------------------------------------------------------------
    # Finalise parameter lib configuration
    versions = params.version.unique()
    version = versions.max()

    if model_name == "sac15":
        model = sac15.SAC15()
        true2trans = sac15.sac15_true2trans
    elif model_name == "gr6j":
        model = gr6j.GR6J()
        true2trans = gr6j.gr6j_true2trans
    elif model_name == "wapaba":
        model = wapaba.WAPABA()
        true2trans = wapaba.wapaba_true2trans
    elif model_name == "ihacres":
        model = ihacres.IHACRES()
        true2trans = ihacres.ihacres_true2trans

    pnames = model.params.names

    plib = params.loc[params.version==version, pnames]
    tplib = plib*0.
    for i, p in plib.iterrows():
        tplib.loc[i, :] = true2trans(p.values)

    means = tplib.mean()
    cov = tplib.cov()

    # Treat SAC15 Sarva separately as it is fixed to 0 in current calib
    if "Sarva" in pnames:
        cov.loc["Sarva", "Sarva"] = 0.1

    # Rounding. Careful with parameters having very small std
    cov_raw = cov.values
    cov = 0*cov_raw
    digits = -np.floor(np.log10(np.diag(cov_raw))-1).astype(int)
    # .. perform rounding starting from low precision to high
    for d in np.unique(digits):
        ii = np.where(digits==d)[0]
        cov[:, ii] = cov_raw[:, ii].round(d)
        cov[ii, :] = cov_raw[ii, :].round(d)

    # Check covar is semidef pos
    np.linalg.cholesky(cov)
    assert np.allclose(cov, cov.T)


    f = flibs / f"paramslib_{model_name}.txt"
    with f.open("w") as fo:
        fo.write(f"--- {model_name} paramslib version {version} ---\n\n")

        txt = " ".join([f"{pn}" for pn in pnames])
        fo.write("Parameter names: \n" + txt + "\n\n")


        defaults = plib.mean(axis=0)
        txt = "\n        ".join([f"{v:0.2f}, #{pn}" \
                        for v, pn in zip(defaults, pnames)])
        parlast = pnames[-1]
        txt = re.sub(", "+parlast, " "+parlast, txt)
        fo.write("Default values: \n[\n        " + txt + "    \n]\n\n")

        mins = plib.min(axis=0)
        txt = "\n        ".join([f"{v:0.2f}, #{pn}" \
                        for v, pn in zip(mins, pnames)])
        txt = re.sub(", "+parlast, " "+parlast, txt)
        fo.write("Min values: \n[\n        " + txt + "    \n]\n\n")

        maxs = plib.max(axis=0)
        txt = "\n        ".join([f"{v:0.2f}, #{pn}" \
                        for v, pn in zip(maxs, pnames)])
        txt = re.sub(", "+parlast, " "+parlast, txt)
        fo.write("Max values: \n[\n        " + txt + "    \n]\n\n")


        txt = ", ".join([f"{v:0.2f}" for v in means.values])
        fo.write("Transformed means: \n[" + txt + "]\n\n")

        txt = ""
        for co in cov:
            line = ", ".join([f"{v}" for v in co])
            txt += "    [" + line + "],\n"
        fo.write("Transformed covariances: \n[\n" + txt + "\n]\n\n")


    # Plot
    for plottype in ["version", "objfun"]:
        plt.close("all")
        if model_name == "sac15":
            nrows, ncols, figsize = 4, 5, (15, 10)
        elif model_name == "gr6j":
            nrows, ncols, figsize = 3, 4, (12, 10)
        elif model_name == "wapaba":
            nrows, ncols, figsize = 3, 3, (10, 10)
        elif model_name == "ihacres":
            nrows, ncols, figsize = 3, 3, (10, 10)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, \
                            constrained_layout=True)

        # Select data to plot
        pat = "|".join([f"^{n}$" for n in pnames]) + "|^nse|bias"
        icols = params.columns.str.findall(pat).astype(bool)
        cols = params.columns[icols]

        if plottype == "version":
            idx = params.siteid.astype(bool)
        else:
            idx = params.version == version

        for iplot, (ax, cn) in enumerate(zip(axs.flat, cols)):
            cc = [cn, "siteid", plottype]
            df = pd.pivot_table(params.loc[idx, cc], \
                            columns=plottype, index="siteid", \
                            values=cn)
            if cn == "bias":
                df = np.abs(df)

            putils.ecdfplot(ax, df)

            if cn == "nse":
                ax.set_xlim((0.2, 0.9))
            elif cn == "nserecip":
                ax.set_xlim((-0.5, 0.5))
            elif cn == "bias":
                ax.set_xlim((0, 0.3))
            else:
                mini = df.min().min()
                maxi = np.abs(df).max().max()
                if maxi>10:
                    if mini > 0:
                        ax.set_xscale("log")
                    else:
                        ax.set_xscale("symlog", linthresh=1)

            if iplot == 0:
                ax.legend(loc=4, fontsize="x-small")
            ax.text(0.05, 0.95, cn, transform=ax.transAxes, fontweight="bold")

        for iax in range(iplot+1, np.prod(axs.shape)):
            axs.flat[iax].axis("off")

        fp = fimg / f"calmodel_{model_name}_v{version}_{plottype}.png"
        fig.savefig(fp)


LOGGER.info("Process completed")

