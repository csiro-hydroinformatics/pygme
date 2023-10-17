#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2023-10-17 Tue 02:07 PM
## Comment : Calibrate the gr4j model
##
## ------------------------------
import sys, re, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from pygme.calibration import ObjFunBCSSE
from pygme.models.gr4j import GR4J, CalibrationGR4J

from hydrodiy.stat import metrics
from hydrodiy.io import iutils

import importlib.util

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Warmup period in days
warmup = 365*6

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent

# Import utils to load test data
spec = importlib.util.spec_from_file_location("testdata", \
            froot.parent / "tests" / "testdata.py")
testdata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(testdata)

#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

# Load test data (see pygme test files)
i = 10
data = testdata.read(f"GR4J_timeseries_{i+1:02d}.csv", \
                        source="output", has_dates=False)
# .. climate inputs
inputs = np.ascontiguousarray(\
                data.loc[:, ["Precip", "PotEvap"]], \
                np.float64)
# .. streamflow data
obs = data.Qobs.values

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
# initialise GR4J
gr = GR4J()

# Define calibration period
LOGGER.info("Define calibration period")
idx_cal = np.arange(len(inputs))>=warmup
idx_cal = idx_cal & (obs >= 0)

# Calibrate
LOGGER.info("Calibrating GR4J")
calib = CalibrationGR4J(ObjFunBCSSE())
calib.workflow(obs, inputs, ical=idx_cal)

params = gr.params.values
LOGGER.info("Calibrated parameters:")
LOGGER.info("\n\t" + "\n\t".join(["X{0} = {1:0.1f}".format(i+1, params[i]) \
                for i in range(4)]))

# Run model
LOGGER.info("Running GR4J and compute stats")
gr = calib.model
gr.inputs = inputs
gr.run()
sim = gr.outputs[:, 0]

# Statistics over calibration period
obsc, simc = obs[idx_cal], sim[idx_cal]
bias = metrics.bias(obsc, simc)
nse = metrics.nse(obsc, simc)
LOGGER.info("\nbias = {0:.3f}\nNSE  = {1:.3f}".format(bias, nse))

